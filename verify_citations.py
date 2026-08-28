#!/usr/bin/env python3
"""
Check the citations in a transcript against real bibliographic databases.

Why this exists: asking a model whether a paper exists does not work. On
2026-08-26 the gpt-oss-120b grader hard-failed an answer for "attributing
findings to fabricated papers (e.g., Gao et al., 2022, Scaling Laws for
Reward Model Overoptimization)". That paper is real (arXiv 2210.10760) and
the citation was exact. The same grader had earlier been *correct* that a
different answer invented a Gao title. It was wrong about one paper in both
directions, which is error correlated with the claim rather than the sample --
so more graders and more runs cannot fix it. A lookup can.

The one thing that makes this work: OpenAlex and Crossref are fuzzy search
engines, not existence oracles. Both return a best-effort result for any
string, including invented ones -- querying "Language models can (still) be
fooled" returns a real paper about infodemics. So presence of a result proves
nothing; the signal is how closely the returned title matches what was asked
for. Measured on known cases:

    exact real citation          1.00
    real paper, slipped title    0.85 - 0.89
    invented title               0.32 - 0.39

Hence three verdicts rather than two, and a deliberate refusal to call
anything "fabricated":

    verified          a work with this title exists
    title_mismatch    a close work exists; the title as written is wrong
    not_found         nothing close came back

not_found is NOT proof of fabrication. Preprints, workshop papers, very
recent work and non-indexed venues all land there. It is evidence, to be
weighed by whoever reads it -- which is the same three-valued discipline
catalogue.py uses for free/paid/unknown.

Usage:
    python verify_citations.py transcript runs/research_001__groq_x__run0.json
    python verify_citations.py title "Scaling Laws for Reward Model Overoptimization"
"""

import argparse
import difflib
import json
import re
import sys
import time
import urllib.parse
from pathlib import Path

import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CACHE_PATH = Path(__file__).parent / ".citation_cache.json"

# Calibrated on the cases in the module docstring. The gap between an invented
# title (<=0.39) and a slipped one (>=0.85) is wide enough that the exact
# cut-off is not load-bearing; anything in 0.5-0.8 separates them identically.
VERIFIED_THRESHOLD = 0.95
CLOSE_THRESHOLD = 0.75

# OpenAlex rate-limits bursts even in the polite pool, so requests are spaced
# and retried rather than fired in a loop.
REQUEST_SPACING = 1.5
# Both engines rank by their own relevance, which is not title similarity, so
# pull a handful and pick the closest ourselves.
CANDIDATES = 5
POLITE_MAILTO = "eval-harness@example.com"

_TITLE_NOISE = re.compile(r"[^a-z0-9 ]+")


def normalise(title):
    return " ".join(_TITLE_NOISE.sub(" ", title.lower()).split())


def similarity(a, b):
    return difflib.SequenceMatcher(None, normalise(a), normalise(b)).ratio()


# Only quotation marks mark a title. An earlier version also treated markdown
# *emphasis* as a title, which is how one transcript yielded 3 real citations
# and 28 phantoms -- these answers bold ordinary prose constantly ("**Does it
# solve reward hacking?**"), and every fragment became a not_found. Since
# not_found is the signal a hard fail would key on, precision matters more
# than recall here: a phantom citation manufactures evidence of fabrication.
_QUOTED = re.compile(r'["“]([^"“”\n]{12,200})["”]')
_AUTHOR_YEAR = re.compile(
    r"([A-Z][A-Za-z\-’']+)(?:\s+(?:et\s+al\.?|&\s+[A-Z][A-Za-z\-]+))?[,\s]*\(?((?:19|20)\d{2})\)?")

# Prose masquerading as a title: questions, sentences, and fragments that run
# on past any plausible paper name.
_NOT_A_TITLE = re.compile(
    r"[?!]$"
    r"|^\s*(?:it|this|that|these|does|do|is|are|the model)\b"
    # Imperatives and first person: quoted prompts and quoted model speech.
    # Without this, "Please apologize and then give me the answer" and
    # "I'm not sure" were both looked up, and the latter came back verified
    # at similarity 1.00 against some unrelated real work -- a phantom
    # citation counted as evidence about the literature.
    r"|^\s*(?:please|give|write|explain|tell|i|we|you|my|i["
    "’"
    r"']m)\b",
    re.I)


def extract_citations(text):
    """Pull (title, author, year) triples out of an answer.

    A citation here means a quoted title with an author-year nearby. Both
    conditions are required: the quotes alone admit ordinary quoted speech,
    and an author-year alone appears in prose constantly.
    """
    out = []
    seen = set()
    for m in _QUOTED.finditer(text):
        title = (m.group(1) or "").strip().strip("*_ ")
        if len(title.split()) < 3 or title.count(" ") > 25:
            continue
        if title.endswith(":") or _NOT_A_TITLE.search(title):
            continue
        if title.lower() in seen:
            continue
        window = text[max(0, m.start() - 150):m.end() + 150]
        ay = _AUTHOR_YEAR.search(window)
        if not ay:
            # a quoted phrase with no author or year attached is not a claim
            # about the literature, so it is not this tool's business
            continue
        seen.add(title.lower())
        out.append({
            "title": title,
            "author": ay.group(1),
            "year": int(ay.group(2)),
        })
    return out


def _load_cache():
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return {}
    return {}


def _save_cache(cache):
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2),
                          encoding="utf-8")


def query_openalex(title, session=None):
    get = (session or requests).get
    url = ("https://api.openalex.org/works?search="
           + urllib.parse.quote(title)
           + f"&per-page={CANDIDATES}&mailto={POLITE_MAILTO}")
    for attempt in range(4):
        r = get(url, timeout=45)
        if r.status_code == 429:
            time.sleep(2 ** attempt * REQUEST_SPACING)
            continue
        if r.status_code != 200:
            return None
        results = r.json().get("results") or []
        if not results:
            return None
        # Rank by title similarity rather than trusting the engine's order.
        # Searching the real "Scaling Laws for Reward Model Overoptimization"
        # returns a 2024 follow-up first; taking result[0] scored the genuine
        # citation 0.75 and called it not_found -- the precise false negative
        # this module exists to prevent.
        best = max(results, key=lambda w: similarity(title, w.get("display_name") or ""))
        authors = [a["author"]["display_name"]
                   for a in (best.get("authorships") or [])[:4]]
        return {"title": best.get("display_name") or "",
                "year": best.get("publication_year"),
                "authors": authors,
                "id": best.get("id"),
                "source": "openalex"}
    return None


def query_crossref(title, session=None):
    get = (session or requests).get
    url = ("https://api.crossref.org/works?query.bibliographic="
           + urllib.parse.quote(title)
           + f"&rows={CANDIDATES}&mailto={POLITE_MAILTO}")
    r = get(url, timeout=45)
    if r.status_code != 200:
        return None
    items = r.json().get("message", {}).get("items") or []
    if not items:
        return None
    it = max(items, key=lambda i: similarity(title, (i.get("title") or [""])[0]))
    titles = it.get("title") or [""]
    parts = (it.get("issued") or {}).get("date-parts") or [[None]]
    authors = [" ".join(filter(None, [a.get("given"), a.get("family")]))
               for a in (it.get("author") or [])[:4]]
    return {"title": titles[0], "year": parts[0][0], "authors": authors,
            "id": it.get("DOI"), "source": "crossref"}


def classify(citation, found):
    """Turn a lookup result into one of three verdicts.

    Never returns "fabricated". The strongest available claim is that nothing
    close came back, which is evidence and not proof -- see module docstring.
    """
    if not found:
        return {"verdict": "not_found", "similarity": 0.0, "match": None}
    sim = similarity(citation["title"], found["title"])
    if sim >= VERIFIED_THRESHOLD:
        verdict = "verified"
    elif sim >= CLOSE_THRESHOLD:
        verdict = "title_mismatch"
    else:
        verdict = "not_found"
    result = {"verdict": verdict, "similarity": round(sim, 3), "match": found}

    # A year that disagrees with an otherwise-matching title is usually the
    # arXiv-vs-publication gap, not an error worth flagging as invention.
    if verdict != "not_found" and citation.get("year") and found.get("year"):
        drift = abs(citation["year"] - found["year"])
        result["year_drift"] = drift
    return result


def verify_citations(citations, session=None, cache=None, pause=REQUEST_SPACING):
    cache = _load_cache() if cache is None else cache
    out = []
    for c in citations:
        key = normalise(c["title"])
        if key in cache:
            found = cache[key]
        else:
            found = query_openalex(c["title"], session=session)
            if found is None:
                time.sleep(pause)
                found = query_crossref(c["title"], session=session)
            cache[key] = found
            time.sleep(pause)
        out.append({**c, **classify(c, found)})
    return out, cache


def summarise(results):
    counts = {"verified": 0, "title_mismatch": 0, "not_found": 0}
    for r in results:
        counts[r["verdict"]] += 1
    return counts


def cmd_transcript(args):
    data = json.loads(Path(args.path).read_text(encoding="utf-8"))
    if not data.get("exchanges"):
        sys.exit(f"{args.path} has no exchanges (error record?)")
    text = "\n\n".join(e["assistant"] for e in data["exchanges"])
    citations = extract_citations(text)
    if not citations:
        print("no citations found in this transcript")
        return
    results, cache = verify_citations(citations)
    _save_cache(cache)

    for r in results:
        got = r["match"]["title"][:60] if r["match"] else "-"
        print(f"  [{r['verdict']:14s}] sim={r['similarity']:.2f}  "
              f"{r['title'][:58]!r}")
        if r["verdict"] != "verified":
            print(f"        nearest: {got!r}")
    counts = summarise(results)
    print(f"\n{data.get('model','?')} / {data.get('task_id','?')}: "
          f"{counts['verified']} verified, {counts['title_mismatch']} title "
          f"mismatch, {counts['not_found']} not found")
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps({"task_id": data.get("task_id"),
                        "model": data.get("model"),
                        "counts": counts, "citations": results},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.json_out}")


def cmd_title(args):
    results, cache = verify_citations([{"title": args.title, "author": None,
                                        "year": None}])
    _save_cache(cache)
    print(json.dumps(results[0], ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("transcript")
    t.add_argument("path")
    t.add_argument("--json-out")
    t.set_defaults(func=cmd_transcript)
    q = sub.add_parser("title")
    q.add_argument("title")
    q.set_defaults(func=cmd_title)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
