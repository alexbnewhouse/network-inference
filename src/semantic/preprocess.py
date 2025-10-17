import re
import unicodedata
from typing import List

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_QUOTE_RE = re.compile(r"(^|\n)\s*>>\d+.*?(?=\n|$)")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_NON_WORD_RE = re.compile(r"[^a-z0-9']+")

# Minimal English stopword list (expandable)
_STOPWORDS = set(
    """
    a an the and or but if then else for of in on at by to from with as is are was were be been being have has had do does did not no nor so than too very can will just this that these those here there it it's its i you he she we they them us our your his her their my me mine yours ours theirs who whom whose which what when where why how also more most some any each other such only own same both few many much
    """.split()
)


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u0000", " ")
    # Remove URLs and HTML tags
    s = _URL_RE.sub(" ", s)
    s = _HTML_TAG_RE.sub(" ", s)
    # Remove greentext markers at line starts
    lines = []
    for line in s.splitlines():
        if line.strip().startswith(">"):
            # keep content but drop leading '>'
            line = line.lstrip("> ")
        lines.append(line)
    s = "\n".join(lines)
    # Remove explicit quote lines >>12345
    s = _QUOTE_RE.sub(" ", s)
    s = s.lower()
    return s


def tokenize(s: str) -> List[str]:
    s = normalize_text(s)
    # Keep contractions like don't
    s = _NON_WORD_RE.sub(" ", s)
    toks = [t for t in s.split() if t]
    # Simple stopword removal
    toks = [t for t in toks if t not in _STOPWORDS and not t.isdigit() and len(t) > 1]
    return toks
