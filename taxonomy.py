"""
Shared bucket definitions.

Both the personal export and the public corpora run through this, otherwise
the two distributions are not comparable and the whole exercise is pointless.

The rules are heuristic and will be wrong on maybe 20-30% of turns. That is
tolerable for a distribution but not for scoring, so hand-check a sample per
bucket before you trust any weighting derived from it. `sample_by_bucket` in
analyze_corpus.py exists for that.
"""

import re

BUCKETS = (
    "code",
    "writing",
    "advisory",
    "factual",
    "roleplay",
    "translation",
    "personal",
    "other",
)

# Order matters. First match wins, so the more specific patterns go first.
RULES = [
    ("code", r"\b(code|function|debug|error|traceback|bug|script|python|javascript|"
             r"sql|regex|api|repo|compile|stack ?trace|syntax)\b"),
    ("translation", r"\b(translate|translation|in (spanish|french|german|hindi|"
                    r"marathi|chinese|japanese|arabic))\b"),
    ("roleplay", r"\b(pretend|role ?play|act as|you are a|imagine you'?re|"
                 r"stay in character)\b"),
    ("writing", r"\b(write|draft|rewrite|edit|summari[sz]e|shorten|proofread|"
                r"essay|email|caption|blog|paraphrase|outline)\b"),
    ("advisory", r"\b(should i|help me decide|worth it|advice|what do you think|"
                 r"which (one|option)|pros and cons|recommend|is it a good idea)\b"),
    ("personal", r"\b(my (partner|boyfriend|girlfriend|husband|wife|mum|mom|dad|"
                 r"family|friend|therapist)|i feel|i'?m anxious|i'?m struggling|"
                 r"relationship)\b"),
    ("factual", r"^(what|who|when|where|why|how)\b.*\?$"),
]

COMPILED = [(name, re.compile(pat, re.IGNORECASE | re.DOTALL)) for name, pat in RULES]


def classify(text):
    if not text or not text.strip():
        return "other"
    t = " ".join(text.split())
    for name, pat in COMPILED:
        if pat.search(t):
            return name
    if t.rstrip().endswith("?"):
        return "factual"
    return "other"


def word_count(text):
    return len(text.split()) if text else 0


def length_band(n):
    if n < 10:
        return "very_short"
    if n < 40:
        return "short"
    if n < 150:
        return "medium"
    return "long"
