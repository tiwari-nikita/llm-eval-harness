#!/usr/bin/env python3
"""
Replay your own real prompts through free models, then pick between the two
answers one keypress at a time. What comes out is a model card: which model
to use for each kind of thing you actually ask, and how sure that is.

Three commands, in order:

    python replay.py sample                      # choose prompts, write a review page
    python replay.py run --approved <file>       # send only what you approved
    python replay.py card <picks file>           # fold in your picks, print the card

Why this exists: the advisory vote never happened. One vote meant reading
~4,500 words and making five judgments, so nobody cast any. Here one vote is
one short prompt, two short answers and one keypress, on prompts taken from
your own history rather than written for the eval.

Privacy is enforced by construction, not by care:
  - `sample` sends nothing anywhere. It writes a local review page.
  - Prompts flagged as possibly private (relationships, health, money, birth
    details, contact details, long pastes) start unticked on that page.
  - `run` refuses to send a prompt, or to use a provider, that is not in the
    approval file the review page downloads, and refuses an approval made for
    a different sample.
  - Everything lives under prompts/replay/, which .gitignore already covers.

Blinding follows make_grading_bundle.py: the pick page holds no model names.
The A/B -> model mapping stays in a separate local state file. Names are not
revealed after each pick either -- after thirty reveals you would learn each
model's house style and start voting on brand.

What this measures, stated plainly: every model gets the same system prompt
asking for a reply under 150 words, so this is the best *short* answer each
model gives to your prompts. That keeps a vote to about a minute. It is a
real constraint on what is measured, like the 350-word one in advisory.yaml.
"""

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

from runner import (DEFAULT_TPM, DailyQuotaExceeded, TokenBudget, call_model,
                    load_providers, safe_slug)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent
REPLAY_DIR = ROOT / "prompts" / "replay"
DEFAULT_EXPORT = ROOT / "prompts" / "raw_export" / "conversations.json"
REVIEW_TEMPLATE = ROOT / "replay_review_template.html"
PICK_TEMPLATE = ROOT / "replay_pick_template.html"

REPLAY_SYSTEM = "Keep your reply under 150 words."
DEFAULT_MAX_TOKENS = 1024  # generous: gemini and gpt-oss spend hidden reasoning tokens first
LONG_WORDS = 400           # above this, a prompt is probably a paste: opt-in only
MAX_WORDS = 1500           # above this, too costly to replay at all

# Order matters: first match wins. Writing tasks come first because there the
# task beats the subject ("turn this into a tweet about BTC" needs a writer,
# not a crypto expert). Then subject areas, where what a model knows matters
# most. Then generic question shapes, and "other" last.
TOPICS = [
    ("social", "Tweets, replies and LinkedIn",
     r"\b(tweets?|twt|twitter|retweet|quote tweet|reply to (this|his|her|their)|"
     r"replies|captions?|linkedin|bio|threads?|reels?|instagram|insta|posts?|"
     r"on x|followers|engagement|impressions|build(ing)? in public)\b"),
    ("writing", "Writing and rewording",
     r"\b(rewrite|re-?phrase|reword|write|draft|edit|proofread|paraphrase|"
     r"complete (it|this)|enhance|improve this|make (it|this) (sound|better|"
     r"shorter|longer|poetic|formal|casual)|synonyms?|other word for|another word|"
     r"poem|song|lyrics|story|haiku|essay|letter|cool ways to say)\b"),
    ("career", "School, career and visa",
     r"\b(sop|statement of purpose|universit(y|ies)|college|admissions?|cv|"
     r"resume|cover letter|jobs?|careers?|interviews?|internships?|co-?op|"
     r"recruit\w*|hiring|case study|assignment|homework|exams?|coursework|"
     r"professor|cohort|ielts|gre|toefl|visa|i-20|f-1|opt|cpt|loan|mpower|"
     r"scholarship|northeastern|mspm|masters?|degree|farewell|resign\w*|"
     r"salary|offer letter|faang|product management)\b"),
    ("crypto", "Crypto and web3",
     r"\b(crypto\w*|web ?3|btc|bitcoin|eth|ethereum|defi|nfts?|tokens?|"
     r"tokeni[sz]ation|ticker|liquidat\w*|gm|bm|ens|on-?chain|coinbase|"
     r"binance|solana|airdrops?|degen|blockchain|dao|memecoins?|altcoins?|"
     r"stablecoins?|hodl|ath)\b"),
    ("astrology", "Astrology, gems and spiritual",
     r"\b(astro\w*|zodiac|horoscope|kundli|vedic|dasha|mahadasha|antardasha|"
     r"yog|nakshatra|lagna|ascendant|rising sign|moon sign|sun sign|"
     r"retrograde|saturn|jupiter|venus|mars|mercury|rahu|ketu|uranus|neptune|"
     r"pluto|aries|taurus|gemini|cancer|leo|virgo|libra|scorpio|sagittarius|"
     r"capricorn|aquarius|pisces|\d+(st|nd|rd|th) house|mc|midheaven|natal|"
     r"birth chart|tarot|numerolog\w*|manifest\w*|crystals?|gem ?stones?|"
     r"stones?|pukhraj|citrine|amethyst|sapphire|neelam|rudraksha|purnima|"
     r"amavasya|omen|karmic|karma|spiritual\w*|chakras?|angel numbers?)\b"),
    ("marketing", "Marketing and business",
     r"\b(marketing|marketer|brand\w*|advertis\w*|campaigns?|go-to-market|gtm|"
     r"startups?|business\w*|customers?|pricing|growth|mrr|arr|revenue|sales|"
     r"pitch|b2b|b2c|merch|neuromarketing|btl|atl|audience|positioning|"
     r"competitors?|market research|saas|app ideas?|moneti[sz]\w*)\b"),
    ("health", "Health, fitness and skin",
     r"\b(health\w*|diet|weight|body fat|gym|workouts?|exercise|cardio|"
     r"weightlifting|fitness|fasting|calories?|protein|creatine|biotin|"
     r"vitamins?|supplements?|skin\w*|acne|tretinoin|retinol|dark circles?|"
     r"hair ?fall|hair loss|sleep|periods?|pcos|pcod|doctor|symptoms?|"
     r"medicine|body odou?r|bloating|mental health|anxiety|therap\w*)\b"),
    ("relationships", "Relationships and feelings",
     r"\b(loml|boyfriend|girlfriend|husband|wife|partner|ex|crush|"
     r"situationship|dating|relationships?|break ?up|forgive|cheat\w*|toxic|"
     r"love|feelings|closure|marriage|ghost(ed|ing))\b"),
    ("tech", "Tech and code",
     r"\b(code|coding|python|javascript|react|html|css|sql|api|bugs?|errors?|"
     r"console|chrome|extensions?|browser|tor|vpn|software|laptop|excel|"
     r"notion|jira|github|scripts?|regex|chatgpt|claude|gpt|llms?)\b"),
    ("advice", "Everyday decisions",
     r"\b(should i|shall i|is it (a )?good idea|good idea|what'?s better|"
     r"which (one|is better)|or should|recommend\w*|worth it|pros and cons|"
     r"better option)\b"),
    ("explain", "Meanings, lookups and explainers",
     r"(\bmeaning\b|\bmeans?\b|\bdefin\w*|\bwhat('?s| is| are| does| do)\b|"
     r"\bwho (is|was)\b|\bexplain\w*|\bwhy (do|does|is|are|did)\b|"
     r"\bhow (do|does|is|to|much|many)\b|\beli5\b|\bsummar\w*|\boverview\b|"
     r"\bdifference between\b|\bhistor(y|ical)\b|\?\s*$)"),
]
OTHER = ("other", "Everything else")
TOPIC_LABELS = dict([(t[0], t[1]) for t in TOPICS] + [OTHER])
_TOPIC_RE = [(key, re.compile(pat, re.I | re.M)) for key, _, pat in TOPICS]
LOOKUP_WORDS = 4  # "roof parapet", "nietzsche": a bare noun phrase is a lookup

# Reasons a prompt starts unticked on the review page. Deliberately broad:
# a false positive costs one click, a false negative sends something private.
PRIVATE_FLAGS = [
    ("relationship", "mentions a relationship",
     r"\b(loml|boyfriend|girlfriend|husband|wife|my partner|my ex|ex'?s|"
     r"crush|situationship|dating|relationships?|break ?up|forgive (him|her)|"
     r"cheat\w*|toxic|closure|texted me|ghosted|wifing)\b"),
    ("health", "mentions health or body",
     r"\b(symptoms?|diagnos\w*|pcos|pcod|periods?|pregnan\w*|medication|"
     r"medicine|doctor|therap\w*|depress\w*|anxiety|mental health|"
     r"body odou?r|acne|my weight|i weigh|body fat|my body)\b|"
     r"\b\d+(\.\d+)?\s?(kgs?|lbs?|pounds)\b"),
    ("money", "mentions money",
     r"\b(loan|salary|debt|bank account|credit score|credit card|net worth|"
     r"income|tax return)\b"),
    ("immigration", "mentions visa or immigration",
     r"\b(visa|passport|i-?20|f-?1|opt|cpt|sevis|green card|h-?1b|immigration)\b"),
    ("birth", "may contain birth details",
     r"\b(born|birth\w*|dob|tob|time of birth|date of birth)\b|"
     r"\b\d{1,2}[:.]\d{2}\s?(am|pm)\b|\b\d{1,2}\s?(am|pm)\b|"
     r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|"
     r"\b\d{1,2}(st|nd|rd|th)?\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s*,?\s*\d{4}\b"),
    ("contact", "may contain contact details",
     r"[\w.+-]+@[\w-]+\.[\w.]+|\b\d{1,5}\s+\w+\s+(street|st|avenue|ave|road|rd|"
     r"blvd|lane|ln|drive|dr)\b"),
]
_PRIVATE_RE = [(k, label, re.compile(p, re.I)) for k, label, p in PRIVATE_FLAGS]
_PHONE_RE = re.compile(r"(?<!\w)\+?\d[\d\s().-]{8,}\d(?!\w)")

# Reasons a prompt cannot be replayed at all, so it never reaches the page.
_NEEDS_CONTEXT_RE = re.compile(
    r"\b(this|these|attached|above|below|my)\s+(images?|pics?|pictures?|photos?|"
    r"screenshots?|videos?|reels?|files?|pdfs?|attachments?|audio|voice notes?)\b|"
    r"\b(this|these|attached|above|below)\s+(links?|docs?|documents?)\b|"
    r"\bi('ve| have)? attached\b|\battached (is|are|here|below|above)\b|"
    r"\bsee (above|below)\b|\b(image|photo|picture|file) i('ll)? (give|send|upload)\b",
    re.I)
_IMAGE_REQUEST_RE = re.compile(
    r"\b(create|generate|make|draw|design)\b[^.\n]{0,25}\b(an? |the )?(images?|"
    r"illustrations?|pictures?|drawings?|logos?|posters?|wallpapers?|avatars?|"
    r"paintings?)\b|\bghibli\b|\bdall-?e\b", re.I)

PROVIDER_NOTES = {
    "google": "Google AI Studio's free tier may use prompts to improve Google "
              "products, and people may review them.",
    "openrouter": "OpenRouter free endpoints pass prompts to the underlying "
                  "provider, whose logging and training terms vary.",
}

TAGS = ["more specific", "right tone", "other was wrong", "other was padded"]
CHOICES = ("a", "b", "tie", "both_bad")


# --------------------------------------------------------------------------
# Reading the export


def active_path(conv):
    """Nodes on the branch the conversation actually ended on, root first.

    A ChatGPT export is a tree: editing a message forks it, and the abandoned
    branch stays in `mapping`. Walking `current_node` back through `parent`
    gives the branch that was really used. Without a current_node, fall back
    to every message in time order, which is right for unedited chats.
    """
    mapping = conv.get("mapping") or {}
    node_id = conv.get("current_node")
    if node_id not in mapping:
        nodes = [n for n in mapping.values() if n.get("message")]
        return sorted(nodes, key=lambda n: n["message"].get("create_time") or 0)
    path, seen = [], set()
    while node_id and node_id in mapping and node_id not in seen:
        seen.add(node_id)
        path.append(mapping[node_id])
        node_id = mapping[node_id].get("parent")
    return path[::-1]


def _user_message(node):
    msg = node.get("message") or {}
    if (msg.get("author") or {}).get("role") != "user":
        return None
    if (msg.get("content") or {}).get("content_type") not in ("text", "multimodal_text"):
        return None
    if (msg.get("metadata") or {}).get("is_visually_hidden_from_conversation"):
        return None
    return msg


def opening_prompts(export):
    """The first thing you typed in each conversation, with what came with it.

    Openings, not later turns: a later turn leans on context the replay
    cannot give ("make it shorter"), while an opening stands on its own.
    """
    out = []
    for conv in export:
        users = [m for m in (_user_message(n) for n in active_path(conv)) if m]
        if not users:
            continue
        first = users[0]
        parts = (first.get("content") or {}).get("parts") or []
        text = "\n".join(p for p in parts if isinstance(p, str)).strip()
        conv_id = conv.get("conversation_id") or conv.get("id") or conv.get("title", "")
        out.append({
            "id": "p_" + hashlib.sha1(str(conv_id).encode("utf-8")).hexdigest()[:10],
            "title": conv.get("title") or "",
            "text": text,
            "words": len(text.split()),
            "n_images": sum(1 for p in parts if isinstance(p, dict)),
            "n_attachments": len((first.get("metadata") or {}).get("attachments") or []),
            "n_turns": len(users),
        })
    return out


# --------------------------------------------------------------------------
# Deciding what a prompt is, and whether it may leave the machine


def _plain(text):
    # phones and ChatGPT's own titles use curly apostrophes, which "what's" misses
    return text.replace("’", "'").replace("‘", "'")


def classify_topic(title, text):
    haystack = _plain(f"{title}\n{text[:400]}")
    for key, pat in _TOPIC_RE:
        if pat.search(haystack):
            return key
    if len(text.split()) <= LOOKUP_WORDS:
        return "explain"
    return OTHER[0]


def unusable_reason(item):
    """Why a prompt cannot be replayed meaningfully, or None."""
    if not item["text"]:
        return "no text"
    if item["n_images"] or item["n_attachments"]:
        return "had an image or file attached"
    if _NEEDS_CONTEXT_RE.search(item["text"]):
        return "refers to something not in the text"
    if _IMAGE_REQUEST_RE.search(item["text"]):
        return "asks for an image"
    if item["words"] > MAX_WORDS:
        return "too long to replay"
    return None


def private_flags(item):
    """Reasons to keep a prompt unticked until you say otherwise."""
    haystack = _plain(f"{item['title']}\n{item['text']}")
    flags = [label for _, label, pat in _PRIVATE_RE if pat.search(haystack)]
    for m in _PHONE_RE.finditer(item["text"]):
        if sum(c.isdigit() for c in m.group()) >= 10:
            flags.append("may contain a phone or ID number")
            break
    if item["words"] > LONG_WORDS:
        flags.append(f"long paste ({item['words']} words)")
    return flags


def allocate(available, total, floor=5):
    """Split `total` across topics in proportion to use, with a floor.

    Proportional, because the topics you use most are the ones the card most
    needs to get right. The floor, because a row resting on three picks is
    not a row. Never gives a topic more than it has, or more than `total`.
    """
    if not available:
        return {}
    floor = min(floor, total // len(available))
    alloc = {t: min(n, floor) for t, n in available.items()}
    remaining = total - sum(alloc.values())
    while remaining > 0:
        open_ = [t for t in available if alloc[t] < available[t]]
        if not open_:
            break
        weight = sum(available[t] for t in open_)
        shares = {t: remaining * available[t] / weight for t in open_}
        given = 0
        for t in open_:
            add = min(available[t] - alloc[t], math.floor(shares[t]))
            alloc[t] += add
            given += add
        # leftovers one at a time, largest remainder first
        for t in sorted(open_, key=lambda t: shares[t] - math.floor(shares[t]), reverse=True):
            if given >= remaining:
                break
            if alloc[t] < available[t]:
                alloc[t] += 1
                given += 1
        if given == 0:
            break
        remaining -= given
    return alloc


def build_sample(openings, n=150, n_optin=30, seed=None):
    """Pick `n` prompts to tick by default and `n_optin` private ones to offer.

    The private ones are sampled from the flagged pool so that opting in is a
    real choice on the page, not a promise with nothing behind it.
    """
    rng = random.Random(seed)
    usable, dropped = [], Counter()
    for item in openings:
        reason = unusable_reason(item)
        if reason:
            dropped[reason] += 1
            continue
        item = dict(item, topic=classify_topic(item["title"], item["text"]),
                    flags=private_flags(item))
        item["default"] = not item["flags"]
        usable.append(item)

    def stratified(pool, k):
        by_topic = defaultdict(list)
        for it in pool:
            by_topic[it["topic"]].append(it)
        alloc = allocate({t: len(v) for t, v in by_topic.items()}, k)
        chosen = []
        for t in sorted(by_topic):
            group = sorted(by_topic[t], key=lambda it: it["id"])
            chosen += rng.sample(group, alloc.get(t, 0))
        return chosen

    clean = [it for it in usable if it["default"]]
    flagged = [it for it in usable if not it["default"]]
    items = stratified(clean, min(n, len(clean))) + stratified(flagged, min(n_optin, len(flagged)))
    items.sort(key=lambda it: (list(TOPIC_LABELS).index(it["topic"]), it["id"]))
    return items, dropped, len(usable)


# --------------------------------------------------------------------------
# Pages


def _inline(template_path, placeholder, payload):
    html = Path(template_path).read_text(encoding="utf-8")
    # a literal "</script>" inside prompt or answer text would close the tag
    blob = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    if placeholder not in html:
        sys.exit(f"{template_path} has no {placeholder} placeholder")
    return html.replace(placeholder, blob)


def provider_summary(providers_cfg, names):
    return [{"name": p, "models": list(providers_cfg[p]["models"]),
             "note": PROVIDER_NOTES.get(p, "")} for p in names]


def cmd_sample(args):
    export_path = Path(args.export)
    if not export_path.exists():
        sys.exit(f"no export at {export_path}. Point --export at your ChatGPT "
                 "conversations.json.")
    providers_cfg = load_providers(args.providers_file)
    names = [p.strip() for p in args.providers.split(",") if p.strip()]
    unknown = [p for p in names if p not in providers_cfg]
    if unknown:
        sys.exit(f"unknown provider(s) {unknown}; check {args.providers_file}")

    export = json.loads(export_path.read_text(encoding="utf-8"))
    openings = opening_prompts(export)
    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(1 << 30)
    items, dropped, n_usable = build_sample(openings, args.n, args.optin, seed)
    sample_id = hashlib.sha1(
        json.dumps([seed, [it["id"] for it in items], names]).encode()).hexdigest()[:12]

    sample = {
        "sample_id": sample_id, "seed": seed,
        "created": datetime.now(timezone.utc).isoformat(),
        "export": str(export_path), "providers": names,
        "system": REPLAY_SYSTEM,
        "topics": TOPIC_LABELS,
        "items": [{k: it[k] for k in ("id", "title", "text", "words", "topic",
                                      "flags", "default")} for it in items],
    }
    REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    (REPLAY_DIR / "sample.json").write_text(
        json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8")

    page = dict(sample, providers=provider_summary(providers_cfg, names))
    page.pop("export")
    out = REPLAY_DIR / "review.html"
    out.write_text(_inline(REVIEW_TEMPLATE, "/*__SAMPLE__*/null", page), encoding="utf-8")

    n_on = sum(1 for it in items if it["default"])
    print(f"{len(openings)} conversations, {n_usable} replayable openings.")
    for reason, c in dropped.most_common():
        print(f"  left out {c:4d}: {reason}")
    print(f"\nSample {sample_id}: {n_on} ticked, {len(items) - n_on} private ones "
          f"offered unticked.")
    for t, c in Counter(it["topic"] for it in items).most_common():
        print(f"  {c:4d}  {TOPIC_LABELS[t]}")
    print(f"\nNothing has been sent anywhere. Open this and review it:\n  {out.resolve()}")
    print("It downloads approved_<id>.json. Then:\n"
          "  python replay.py run --approved <that file> --dry-run\n"
          "  python replay.py run --approved <that file>")


# --------------------------------------------------------------------------
# Approval and running


def load_env_file(path):
    """Fill missing API-key env vars from a local .env, never overriding."""
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        os.environ.setdefault(key, value.strip().strip('"').strip("'"))


def load_approval(sample, approved_path):
    """The approved subset of the sample, or exit saying exactly why not."""
    approval = json.loads(Path(approved_path).read_text(encoding="utf-8"))
    if approval.get("sample_id") != sample["sample_id"]:
        sys.exit(f"{approved_path} approves sample {approval.get('sample_id')!r}, "
                 f"but the current sample is {sample['sample_id']!r}. Review the "
                 "current prompts/replay/review.html and download a fresh approval.")
    by_id = {it["id"]: it for it in sample["items"]}
    include = approval.get("include") or []
    stray = [i for i in include if i not in by_id]
    if stray:
        sys.exit(f"{approved_path} lists {len(stray)} prompt id(s) that are not in "
                 "this sample; refusing to guess what they are.")
    providers = approval.get("providers") or []
    extra = [p for p in providers if p not in sample["providers"]]
    if extra:
        sys.exit(f"{approved_path} names provider(s) {extra} that the review page "
                 "never showed; refusing.")
    if not include:
        sys.exit(f"{approved_path} approves no prompts.")
    topics = approval.get("topics") or {}
    items = []
    for i in include:
        it = dict(by_id[i])
        if topics.get(i) in TOPIC_LABELS:
            it["topic"] = topics[i]
        items.append(it)
    return items, providers


def model_list(providers_cfg, providers):
    return [(p, m) for p in providers for m in providers_cfg[p]["models"]]


def assign_pairs(items, models, seed):
    """Give each prompt one pair of models, balanced overall and per topic.

    Every model does not answer every prompt: that would double the quota for
    no extra votes. One continuous rotation through the pairs, walked topic
    by topic, keeps every pair's count within one of every other's both in
    each topic and overall.
    """
    pairs = list(combinations(models, 2))
    if not pairs:
        sys.exit("need at least two models to compare")
    rng = random.Random(f"{seed}:pairs")
    rng.shuffle(pairs)
    by_topic = defaultdict(list)
    for it in sorted(items, key=lambda it: it["id"]):
        by_topic[it["topic"]].append(it)
    out, k = {}, 0
    for topic in sorted(by_topic):
        group = by_topic[topic]
        rng.shuffle(group)
        for it in group:
            out[it["id"]] = pairs[k % len(pairs)]
            k += 1
    return out


def answer_path(item_id, provider, model):
    return REPLAY_DIR / "answers" / f"{item_id}__{provider}_{safe_slug(model)}.json"


def healthy_answer(path):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if data.get("content", "").strip() else None


def cmd_run(args):
    sample_path = REPLAY_DIR / "sample.json"
    if not sample_path.exists():
        sys.exit("no sample yet. Run: python replay.py sample")
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    items, providers = load_approval(sample, args.approved)
    providers_cfg = load_providers(args.providers_file)
    models = model_list(providers_cfg, providers)
    pairs = assign_pairs(items, models, sample["seed"])

    jobs = [(it, p, m) for it in items for (p, m) in pairs[it["id"]]]
    todo = [j for j in jobs if not healthy_answer(answer_path(j[0]["id"], j[1], j[2]))]
    per_model = Counter(f"{p}/{m}" for _, p, m in todo)
    est = sum(len(it["text"]) // 4 + 60 + 250 for it, _, _ in todo)

    print(f"{len(items)} approved prompts, {len(models)} models, "
          f"{len(jobs)} answers needed, {len(jobs) - len(todo)} already saved.")
    for key, c in sorted(per_model.items()):
        print(f"  {c:4d} calls  {key}")
    print(f"  ~{est:,} tokens in total, roughly")
    print(f"System prompt, identical for every model: {sample['system']!r}")

    if args.dry_run:
        print("\nDry run: nothing sent.")
        return

    # copied only on a real run, so the record matches what was actually sent
    REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    (REPLAY_DIR / "approved.json").write_text(
        Path(args.approved).read_text(encoding="utf-8"), encoding="utf-8")

    load_env_file(ROOT / ".env")
    keys = {p: os.environ.get(providers_cfg[p]["api_key_env"]) for p in providers}
    missing = [providers_cfg[p]["api_key_env"] for p, k in keys.items() if not k]
    if missing and todo:
        # dropping a provider would silently change which pairs get compared
        sys.exit(f"missing API key(s): {', '.join(missing)}. Set them or add them "
                 "to .env; the approved providers must all be runnable.")

    budgets = {p: TokenBudget(args.tpm) for p in providers}
    exhausted = set()
    (REPLAY_DIR / "answers").mkdir(parents=True, exist_ok=True)
    for n, (it, p, m) in enumerate(todo, 1):
        if (p, m) in exhausted:
            continue
        print(f"[{n}/{len(todo)}] {it['id']} x {p}/{m}")
        messages = [{"role": "system", "content": sample["system"]},
                    {"role": "user", "content": it["text"]}]
        try:
            reply = call_model(providers_cfg[p]["base_url"], keys[p], m, messages,
                               max_tokens=args.max_tokens, budget=budgets[p])
            record = {"item_id": it["id"], "provider": p, "model": m,
                      "content": reply["content"],
                      "finish_reason": reply["finish_reason"],
                      "truncated": reply["finish_reason"] == "length",
                      "completion_tokens": reply["completion_tokens"],
                      "max_tokens": args.max_tokens, "system": sample["system"],
                      "timestamp": datetime.now(timezone.utc).isoformat()}
        except DailyQuotaExceeded as e:
            # one model's day being gone is no reason to stop the others
            print(f"    {p}/{m}: daily quota gone, skipping its remaining calls "
                  f"({str(e)[:120]})", file=sys.stderr)
            exhausted.add((p, m))
            continue
        except Exception as e:
            print(f"    FAILED: {e}", file=sys.stderr)
            record = {"item_id": it["id"], "provider": p, "model": m,
                      "error": str(e),
                      "timestamp": datetime.now(timezone.utc).isoformat()}
        answer_path(it["id"], p, m).write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        time.sleep(args.gap)

    build_pick_page(sample, items, pairs)


def build_pick_page(sample, items, pairs):
    """Write pick.html plus the mapping it must never contain.

    The bundle id is derived from the sample and the pairing, so re-running
    after a quota stop rebuilds the *same* page with more prompts in it, and
    picks already saved in the browser still line up.
    """
    pair_key = json.dumps(sorted((i, list(map(list, pr))) for i, pr in pairs.items()))
    bundle_id = hashlib.sha1(f"{sample['sample_id']}|{pair_key}".encode()).hexdigest()[:12]

    shown, mapping, missing = [], {}, 0
    for it in items:
        (p1, m1), (p2, m2) = pairs[it["id"]]
        ans1 = healthy_answer(answer_path(it["id"], p1, m1))
        ans2 = healthy_answer(answer_path(it["id"], p2, m2))
        if not (ans1 and ans2):
            missing += 1
            continue
        sides = [((p1, m1), ans1), ((p2, m2), ans2)]
        if random.Random(f"{sample['seed']}:{it['id']}:sides").random() < 0.5:
            sides.reverse()
        (pa, ma), a = sides[0]
        (pb, mb), b = sides[1]
        shown.append({"id": it["id"], "prompt": it["text"],
                      "a": a["content"], "b": b["content"],
                      "a_cut": bool(a.get("truncated")), "b_cut": bool(b.get("truncated"))})
        mapping[it["id"]] = {
            "topic": it["topic"],
            "a": {"provider": pa, "model": ma, "words": len(a["content"].split())},
            "b": {"provider": pb, "model": mb, "words": len(b["content"].split())},
        }

    rng = random.Random(f"{sample['seed']}:order")
    rng.shuffle(shown)  # so topics are interleaved and fatigue is spread evenly
    state = {"bundle_id": bundle_id, "sample_id": sample["sample_id"],
             "created": datetime.now(timezone.utc).isoformat(), "mapping": mapping}
    (REPLAY_DIR / f"pick_state_{bundle_id}.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    page = {"bundle_id": bundle_id, "tags": TAGS, "items": shown}
    out = REPLAY_DIR / "pick.html"
    out.write_text(_inline(PICK_TEMPLATE, "/*__BUNDLE__*/null", page), encoding="utf-8")

    print(f"\n{len(shown)} prompts ready to pick"
          + (f", {missing} still waiting on an answer (re-run to fill them in)" if missing else "")
          + f".\nOpen: {out.resolve()}")
    print("Your progress saves as you go. When you stop, download your picks, then:\n"
          "  python replay.py card <downloaded picks file>")
    return out


# --------------------------------------------------------------------------
# The card


def short_name(model):
    return model.split("/")[-1].replace(":free", "")


def ingest(paths):
    """Merge downloaded picks files into prompts/replay/picks.json."""
    store_path = REPLAY_DIR / "picks.json"
    store = json.loads(store_path.read_text(encoding="utf-8")) if store_path.exists() else {}
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if "bundle_id" not in payload or "picks" not in payload:
            sys.exit(f"{path}: not a picks file")
        state_path = REPLAY_DIR / f"pick_state_{payload['bundle_id']}.json"
        if not state_path.exists():
            sys.exit(f"{path}: no local mapping for bundle {payload['bundle_id']}, so "
                     "these picks cannot be matched to models.")
        mapping = json.loads(state_path.read_text(encoding="utf-8"))["mapping"]
        n = 0
        for item_id, pick in payload["picks"].items():
            if item_id not in mapping or pick.get("choice") not in CHOICES:
                continue
            side = mapping[item_id]
            store[item_id] = {
                "bundle_id": payload["bundle_id"], "topic": side["topic"],
                "model_a": f"{side['a']['provider']}/{side['a']['model']}",
                "model_b": f"{side['b']['provider']}/{side['b']['model']}",
                "a_words": side["a"]["words"], "b_words": side["b"]["words"],
                "choice": pick["choice"], "tags": pick.get("tags") or [],
                "at": pick.get("at"), "ms": pick.get("ms"),
            }
            n += 1
        print(f"{Path(path).name}: {n} pick(s) folded in")
    REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    store_path.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
    return store


def bradley_terry(votes, models, iters=300):
    """Strength per model from pairwise outcomes, ties counted as half a win.

    Each pair of models also gets one virtual tie, a weak prior that keeps a
    model with two lucky wins from reading as certain, and keeps the fit
    defined for a model that never won.
    """
    idx = {m: i for i, m in enumerate(models)}
    k = len(models)
    wins = [0.5 * (k - 1)] * k                     # the virtual ties
    games = [[0.0 if i == j else 1.0 for j in range(k)] for i in range(k)]
    for a, b, s in votes:  # s = score for a: 1, 0.5 or 0
        i, j = idx[a], idx[b]
        wins[i] += s
        wins[j] += 1 - s
        games[i][j] += 1
        games[j][i] += 1
    p = [1.0] * k
    for _ in range(iters):  # Hunter's MM algorithm
        new = [wins[i] / sum(games[i][j] / (p[i] + p[j]) for j in range(k) if j != i)
               for i in range(k)]
        g = math.exp(sum(math.log(v) for v in new) / k)
        new = [v / g for v in new]
        done = max(abs(x - y) for x, y in zip(new, p)) < 1e-9
        p = new
        if done:
            break
    return dict(zip(models, p))


def topic_verdict(records, models, n_boot=400, seed=0):
    score = {"a": 1.0, "b": 0.0, "tie": 0.5, "both_bad": 0.5}
    votes = [(r["model_a"], r["model_b"], score[r["choice"]]) for r in records]
    decisive = sum(1 for r in records if r["choice"] in ("a", "b"))
    undecided = len(records) - decisive
    strengths = bradley_terry(votes, models)
    leader = max(models, key=lambda m: strengths[m])
    rng = random.Random(seed)
    hits = 0
    boots = n_boot if decisive >= 6 else 0  # below that the verdict is already known
    for _ in range(boots):
        boot = [votes[rng.randrange(len(votes))] for _ in votes]
        s = bradley_terry(boot, models, iters=100)
        hits += max(models, key=lambda m: s[m]) == leader
    conf = hits / boots if boots else 0.0
    if decisive < 6:
        verdict = "too few picks"
    elif conf >= 0.9:
        verdict = "clear"
    elif conf >= 0.7:
        verdict = "leaning"
    else:
        verdict = "can't tell yet"
    return {"leader": leader, "confidence": conf, "verdict": verdict,
            "n": len(records), "undecided": undecided, "strengths": strengths}


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def bias_checks(records):
    """How your own picks lean: toward the left side, toward the longer answer."""
    decisive = [r for r in records if r["choice"] in ("a", "b")]
    left = sum(1 for r in decisive if r["choice"] == "a")
    sized = [r for r in decisive if r["a_words"] != r["b_words"]]
    longer = sum(1 for r in sized
                 if (r["choice"] == "a") == (r["a_words"] > r["b_words"]))
    return {"decisive": len(decisive), "left": left,
            "sized": len(sized), "longer": longer,
            "both_bad": sum(1 for r in records if r["choice"] == "both_bad")}


def render_card(store):
    records = list(store.values())
    if not records:
        return "No picks yet. Open prompts/replay/pick.html and start picking."
    models = sorted({r["model_a"] for r in records} | {r["model_b"] for r in records})
    by_topic = defaultdict(list)
    for r in records:
        by_topic[r["topic"]].append(r)

    rows = []
    for t in sorted(by_topic, key=lambda t: -len(by_topic[t])):
        rows.append((TOPIC_LABELS.get(t, t), topic_verdict(by_topic[t], models)))
    rows.append(("Everything", topic_verdict(records, models)))

    lines = [f"Your model card  ·  {len(records)} picks  ·  "
             f"{datetime.now().strftime('%Y-%m-%d')}", ""]
    lines.append(f"  {'Topic':32}{'Pick this':24}{'How sure':22}{'Picks':>6}{'Undecided':>11}")
    for label, v in rows:
        if label == "Everything":
            lines.append("  " + "-" * 93)
        sure = v["verdict"]
        if v["verdict"] in ("clear", "leaning", "can't tell yet"):
            sure += f" ({v['confidence']:.0%})"
        pick = short_name(v["leader"]) if v["verdict"] in ("clear", "leaning") else "-"
        lines.append(f"  {label:32}{pick:24}{sure:22}{v['n']:>6}"
                     f"{v['undecided'] / v['n']:>10.0%}")

    b = bias_checks(records)
    lines += ["", "  Checks on you"]
    if b["decisive"]:
        lo, hi = wilson(b["left"], b["decisive"])
        lean = "you favour the left" if lo > 0.5 else "you favour the right" if hi < 0.5 else "fine"
        lines.append(f"    picked the left answer    {b['left'] / b['decisive']:.0%} of "
                     f"{b['decisive']}  (95% CI {lo:.0%}-{hi:.0%})  {lean}")
    if b["sized"]:
        lo, hi = wilson(b["longer"], b["sized"])
        lean = ("you favour longer answers" if lo > 0.5
                else "you favour shorter answers" if hi < 0.5 else "no length lean")
        lines.append(f"    picked the longer answer  {b['longer'] / b['sized']:.0%} of "
                     f"{b['sized']}  (95% CI {lo:.0%}-{hi:.0%})  {lean}")
    lines.append(f"    both bad                  {b['both_bad']} pick(s)")
    tag_counts = Counter(t for r in records for t in r["tags"])
    if tag_counts:
        lines.append("    reasons you gave          "
                     + ", ".join(f"{t} {c}" for t, c in tag_counts.most_common()))
    lines += ["",
              "  How sure = share of 400 resamples of your picks in which the same",
              "  model comes out on top. Undecided = ties plus both-bad. A topic that",
              "  is mostly undecided is an answer too: there, any of these will do.",
              "  N of 1, short answers only (under 150 words), models as of this run."]
    return "\n".join(lines)


def cmd_card(args):
    store = ingest(args.paths) if args.paths else None
    if store is None:
        path = REPLAY_DIR / "picks.json"
        store = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    print(render_card(store))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample", help="choose prompts and write the review page")
    s.add_argument("--export", default=str(DEFAULT_EXPORT))
    s.add_argument("--providers", default="groq,google",
                   help="who would receive the prompts; shown on the review page")
    s.add_argument("--providers-file", default=str(ROOT / "providers.yaml"))
    s.add_argument("-n", type=int, default=150, help="prompts ticked by default")
    s.add_argument("--optin", type=int, default=30,
                   help="flagged prompts offered unticked")
    s.add_argument("--seed", type=int)
    s.set_defaults(func=cmd_sample)

    r = sub.add_parser("run", help="send approved prompts and build the pick page")
    r.add_argument("--approved", required=True, help="file downloaded from review.html")
    r.add_argument("--providers-file", default=str(ROOT / "providers.yaml"))
    r.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    r.add_argument("--tpm", type=int, default=DEFAULT_TPM)
    r.add_argument("--gap", type=float, default=1.0, help="seconds between calls")
    r.add_argument("--dry-run", action="store_true", help="count and estimate, send nothing")
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("card", help="fold in picks and print your model card")
    c.add_argument("paths", nargs="*", help="picks files downloaded from pick.html")
    c.set_defaults(func=cmd_card)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
