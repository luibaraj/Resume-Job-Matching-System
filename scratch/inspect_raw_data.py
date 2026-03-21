#!/usr/bin/env python3
"""
Inspect raw Greenhouse job data for quality issues.

Samples 500 jobs and checks:
  - Completeness: missing / empty / near-empty descriptions
  - Encoding:     control chars, mojibake, high-unicode after decode
  - HTML format:  entity-escaped HTML (Greenhouse's native format),
                  unescaped raw HTML, HTML entities left after decoding,
                  specific tags of interest (iframe, img, <a>)
  - Other formats: JSON-like, XML, markdown
  - Embedded content: bare URLs in text, iframes, images
  - Field nulls: title, location, department, job_type, company_name
"""

import html
import os
import re
import sqlite3
import unicodedata
from collections import Counter
from pathlib import Path
from dotenv import load_dotenv

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DB_PATH = os.getenv("DB_PATH", "../data/jobs.db")
SAMPLE_SIZE = 500

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------
# Escaped HTML tags as stored in DB: &lt;div ... &gt;
ESCAPED_HTML_TAG = re.compile(
    r'&lt;(?P<tag>[a-zA-Z][a-zA-Z0-9]*)(?:[^&]|&(?!gt;))*?(?:/?&gt;|&gt;)',
    re.IGNORECASE,
)
# Raw (unescaped) HTML tags
RAW_HTML_TAG = re.compile(r'<[a-zA-Z][a-zA-Z0-9]*\b[^>]*>', re.IGNORECASE)
# Any HTML entity (named or numeric)
HTML_ENTITY = re.compile(r'&(?:[a-zA-Z]+|#\d+|#x[0-9a-fA-F]+);')
# Specifically double-encoded structural entities (&lt; &gt; &quot; &amp; &apos;)
STRUCTURAL_ENTITY = re.compile(r'&(?:lt|gt|quot|apos|amp);', re.IGNORECASE)
# Non-breaking / whitespace entities that survive decoding
NBSP_PATTERN = re.compile(r'&nbsp;|\\u00a0|\xa0', re.IGNORECASE)
# Mojibake patterns (UTF-8 bytes read as Latin-1)
MOJIBAKE = re.compile(r'â€[™""œ]|â€"|Â[»«]|Ã[©àèêâ]', re.IGNORECASE)
# Control chars (excluding \t \n \r)
CONTROL_CHARS = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
# Bare URLs in text
URL_PATTERN = re.compile(r'https?://[^\s<>"\']+')
# Iframes and images in the raw stored string
IFRAME_PATTERN = re.compile(r'&lt;iframe|<iframe', re.IGNORECASE)
IMG_PATTERN = re.compile(r'&lt;img|<img', re.IGNORECASE)
# JSON-like (starts with { or [)
JSON_LIKE = re.compile(r'^\s*[{\[]', re.MULTILINE)
# XML declaration / CDATA
XML_PATTERN = re.compile(r'<\?xml|<!\[CDATA', re.IGNORECASE)
# Markdown signals
MD_HEADER = re.compile(r'^#{1,6}\s+', re.MULTILINE)
MD_FENCE = re.compile(r'```')
MD_BOLD_ITALIC = re.compile(r'(?<!\w)[*_]{1,3}\S')

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def decode_greenhouse(text: str) -> str:
    """Decode Greenhouse's double-entity-encoded HTML to raw HTML, then strip tags."""
    decoded = html.unescape(html.unescape(text))
    if HAS_BS4:
        return BeautifulSoup(decoded, "html.parser").get_text(separator=" ")
    # Fallback: strip tags with regex
    return re.sub(r'<[^>]+>', ' ', decoded)


def pct(n: int, total: int) -> str:
    return f"{n / total * 100:.1f}%" if total else "n/a"


def section(title: str) -> None:
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print('=' * 80)


def subsection(title: str) -> None:
    print(f"\n  --- {title} ---")

# ---------------------------------------------------------------------------
# Connect & sample
# ---------------------------------------------------------------------------
conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

cur.execute("SELECT COUNT(*) FROM jobs")
total_db = cur.fetchone()[0]

cur.execute(
    "SELECT id, title, location, description, department, job_type, company_name, board_token "
    "FROM jobs ORDER BY RANDOM() LIMIT ?",
    (SAMPLE_SIZE,),
)
jobs = cur.fetchall()
N = len(jobs)

print(f"Database total: {total_db} jobs")
print(f"Sample size:    {N} jobs")
if not HAS_BS4:
    print("  [!] beautifulsoup4 not installed – using regex tag stripper for plain-text checks")

# ---------------------------------------------------------------------------
# Per-job analysis
# ---------------------------------------------------------------------------
# Counters
counts: dict[str, int] = Counter()
# Detail lists for examples
examples: dict[str, list] = {k: [] for k in [
    "missing_or_empty", "very_short_text", "control_chars", "mojibake",
    "high_unicode", "structural_entities", "escaped_html", "raw_html",
    "nbsp_survived", "iframe", "img", "url_in_text", "json", "xml", "markdown",
]}

# Field-null tracking
field_nulls: dict[str, int] = Counter()
NULLABLE_FIELDS = ["title", "location", "department", "job_type", "company_name"]

# Entity / tag counters for the whole sample
all_escaped_tags: Counter = Counter()
all_entities: Counter = Counter()

for job in jobs:
    jid = job["id"]
    title = job["title"] or ""

    # ----- field-null check -----
    for field in NULLABLE_FIELDS:
        if not job[field]:
            field_nulls[field] += 1

    # ----- description completeness -----
    desc = job["description"]
    if desc is None:
        counts["missing_null"] += 1
        examples["missing_or_empty"].append({"id": jid, "title": title, "reason": "NULL"})
        continue
    if desc.strip() == "":
        counts["missing_empty"] += 1
        examples["missing_or_empty"].append({"id": jid, "title": title, "reason": "empty string"})
        continue

    # ----- control characters -----
    ctrl = CONTROL_CHARS.findall(desc)
    if ctrl:
        counts["control_chars"] += 1
        examples["control_chars"].append({"id": jid, "title": title,
                                           "chars": sorted(set(repr(c) for c in ctrl)),
                                           "count": len(ctrl)})

    # ----- mojibake -----
    if MOJIBAKE.search(desc):
        counts["mojibake"] += 1
        m = MOJIBAKE.search(desc)
        examples["mojibake"].append({"id": jid, "title": title,
                                      "snippet": desc[max(0, m.start()-10):m.end()+20]})

    # ----- structural HTML entities (double-encoded) -----
    ent_matches = STRUCTURAL_ENTITY.findall(desc)
    if ent_matches:
        counts["structural_entities"] += 1
        all_entities.update(e.lower() for e in ent_matches)
        examples["structural_entities"].append({"id": jid, "title": title,
                                                 "count": len(ent_matches),
                                                 "unique": sorted(set(e.lower() for e in ent_matches))})

    # ----- escaped HTML tag inventory -----
    esc_tags = ESCAPED_HTML_TAG.findall(desc)
    if esc_tags:
        counts["escaped_html"] += 1
        all_escaped_tags.update(t.lower() for t in esc_tags)
        if len(examples["escaped_html"]) < 5:
            examples["escaped_html"].append({"id": jid, "title": title,
                                              "snippet": desc[:120]})

    # ----- raw (unescaped) HTML tags -----
    if RAW_HTML_TAG.search(desc):
        counts["raw_html"] += 1
        examples["raw_html"].append({"id": jid, "title": title, "snippet": desc[:120]})

    # ----- any HTML entity -----
    if HTML_ENTITY.search(desc):
        counts["any_entity"] += 1

    # ----- iframes -----
    if IFRAME_PATTERN.search(desc):
        counts["iframe"] += 1
        examples["iframe"].append({"id": jid, "title": title})

    # ----- images -----
    if IMG_PATTERN.search(desc):
        counts["img"] += 1
        examples["img"].append({"id": jid, "title": title})

    # ----- plain-text analysis (after decoding & stripping) -----
    plain = decode_greenhouse(desc)

    # very short plain text
    if len(plain.strip()) < 100:
        counts["very_short_text"] += 1
        examples["very_short_text"].append({"id": jid, "title": title,
                                             "text": plain.strip()[:120]})

    # &nbsp; / \xa0 surviving into plain text
    if NBSP_PATTERN.search(plain):
        counts["nbsp_survived"] += 1
        if len(examples["nbsp_survived"]) < 5:
            examples["nbsp_survived"].append({"id": jid, "title": title,
                                               "snippet": plain[:120]})

    # high-unicode chars after decode (punctuation, symbols, etc.)
    unusual = [c for c in plain
               if ord(c) > 127
               and unicodedata.category(c) in ('Po', 'Pd', 'Pi', 'Pf', 'Ps', 'Pe', 'So', 'Sm', 'Sk')]
    if unusual:
        counts["high_unicode"] += 1
        if len(examples["high_unicode"]) < 5:
            examples["high_unicode"].append({"id": jid, "title": title,
                                              "chars": sorted(set(unusual))[:10]})

    # bare URLs in plain text
    urls = URL_PATTERN.findall(plain)
    if urls:
        counts["url_in_text"] += 1
        if len(examples["url_in_text"]) < 5:
            examples["url_in_text"].append({"id": jid, "title": title,
                                             "urls": urls[:3]})

    # JSON-like
    if JSON_LIKE.search(desc):
        counts["json"] += 1
        examples["json"].append({"id": jid, "title": title, "snippet": desc[:100]})

    # XML declaration
    if XML_PATTERN.search(desc):
        counts["xml"] += 1
        examples["xml"].append({"id": jid, "title": title})

    # Markdown signals (in plain text, more reliable)
    if MD_HEADER.search(plain) or MD_FENCE.search(plain):
        counts["markdown"] += 1
        examples["markdown"].append({"id": jid, "title": title})

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
section("RAW GREENHOUSE DATA QUALITY ANALYSIS")
print(f"  Sample: {N} of {total_db} total jobs  |  bs4 available: {HAS_BS4}")

# ---- 1. Completeness ----
section("1. COMPLETENESS")
total_bad = counts["missing_null"] + counts["missing_empty"]
print(f"  NULL descriptions:          {counts['missing_null']:>4}  ({pct(counts['missing_null'], N)})")
print(f"  Empty-string descriptions:  {counts['missing_empty']:>4}  ({pct(counts['missing_empty'], N)})")
print(f"  Very short after decoding:  {counts['very_short_text']:>4}  ({pct(counts['very_short_text'], N)})  (<100 chars of text)")
print(f"  Total unusable:             {total_bad:>4}  ({pct(total_bad, N)})")

subsection("Field nulls across ALL sampled jobs")
for field in NULLABLE_FIELDS:
    print(f"  {field:<20} {field_nulls[field]:>4} NULL/empty  ({pct(field_nulls[field], N)})")

if examples["missing_or_empty"]:
    subsection("Missing/Empty examples")
    for e in examples["missing_or_empty"][:8]:
        print(f"    Job {e['id']:>5}: {e['title'][:55]}  ({e['reason']})")

if examples["very_short_text"]:
    subsection("Near-empty after decoding")
    for e in examples["very_short_text"][:5]:
        print(f"    Job {e['id']:>5}: {e['title'][:55]}")
        print(f"              text: {repr(e['text'])}")

# ---- 2. Encoding ----
section("2. ENCODING & CHARACTER ISSUES")
print(f"  Control characters:         {counts['control_chars']:>4}  ({pct(counts['control_chars'], N)})")
print(f"  Mojibake detected:          {counts['mojibake']:>4}  ({pct(counts['mojibake'], N)})")
print(f"  High-unicode after decode:  {counts['high_unicode']:>4}  ({pct(counts['high_unicode'], N)})  (curly quotes, em-dash, §, …)")
print(f"  &nbsp; surviving decode:    {counts['nbsp_survived']:>4}  ({pct(counts['nbsp_survived'], N)})")

if examples["control_chars"]:
    subsection("Control char examples")
    for e in examples["control_chars"][:5]:
        print(f"    Job {e['id']:>5}: {e['title'][:55]}  chars={e['chars']}  n={e['count']}")

if examples["mojibake"]:
    subsection("Mojibake examples")
    for e in examples["mojibake"][:5]:
        print(f"    Job {e['id']:>5}: {e['title'][:55]}")
        print(f"              snippet: {repr(e['snippet'])}")

if examples["high_unicode"]:
    subsection("High-unicode char examples")
    for e in examples["high_unicode"][:5]:
        print(f"    Job {e['id']:>5}: {e['title'][:55]}  chars={e['chars']}")

if examples["nbsp_survived"]:
    subsection("&nbsp; surviving into plain text")
    for e in examples["nbsp_survived"][:3]:
        print(f"    Job {e['id']:>5}: {e['title'][:55]}")
        print(f"              snippet: {repr(e['snippet'][:80])}")

# ---- 3. HTML format ----
section("3. HTML STRUCTURE")
print(f"  Escaped HTML (&lt;tag&gt;):  {counts['escaped_html']:>4}  ({pct(counts['escaped_html'], N)})  — Greenhouse native format")
print(f"  Any structural entity:      {counts['structural_entities']:>4}  ({pct(counts['structural_entities'], N)})  (&lt; &gt; &quot; &amp; &apos;)")
print(f"  Any HTML entity:            {counts['any_entity']:>4}  ({pct(counts['any_entity'], N)})  (&nbsp; etc. included)")
print(f"  Raw (unescaped) HTML tags:  {counts['raw_html']:>4}  ({pct(counts['raw_html'], N)})  — unexpected if DB stores escaped")
print(f"  Iframes:                    {counts['iframe']:>4}  ({pct(counts['iframe'], N)})")
print(f"  Images (<img>):             {counts['img']:>4}  ({pct(counts['img'], N)})")

if all_escaped_tags:
    subsection("Escaped HTML tag inventory (top 20)")
    for tag, cnt in all_escaped_tags.most_common(20):
        print(f"    <{tag}>: {cnt}")

if examples["raw_html"]:
    subsection("Raw HTML tag examples (unexpected)")
    for e in examples["raw_html"][:5]:
        print(f"    Job {e['id']:>5}: {e['title'][:55]}")
        print(f"              snippet: {repr(e['snippet'][:80])}")

if examples["iframe"]:
    subsection("Iframe jobs")
    for e in examples["iframe"]:
        print(f"    Job {e['id']:>5}: {e['title'][:60]}")

if examples["img"]:
    subsection("Image jobs (sample)")
    for e in examples["img"][:5]:
        print(f"    Job {e['id']:>5}: {e['title'][:60]}")

# ---- 4. Embedded content ----
section("4. EMBEDDED CONTENT IN PLAIN TEXT")
print(f"  Bare URLs:                  {counts['url_in_text']:>4}  ({pct(counts['url_in_text'], N)})")

if examples["url_in_text"]:
    subsection("URL examples")
    for e in examples["url_in_text"][:5]:
        print(f"    Job {e['id']:>5}: {e['title'][:55]}")
        for u in e["urls"]:
            print(f"              {u[:90]}")

# ---- 5. Other formats ----
section("5. OTHER FORMAT SIGNALS")
print(f"  JSON-like patterns:         {counts['json']:>4}  ({pct(counts['json'], N)})")
print(f"  XML declaration/CDATA:      {counts['xml']:>4}  ({pct(counts['xml'], N)})")
print(f"  Markdown (headers/fences):  {counts['markdown']:>4}  ({pct(counts['markdown'], N)})")

if examples["json"]:
    subsection("JSON examples")
    for e in examples["json"][:3]:
        print(f"    Job {e['id']:>5}: {e['title'][:55]}")
        print(f"              {repr(e['snippet'][:80])}")

# ---- 6. Summary ----
section("6. SUMMARY & PREPROCESSING NOTES")

ok = counts["escaped_html"] == N and counts["raw_html"] == 0
n_ctrl    = counts["control_chars"]
n_moji    = counts["mojibake"]
n_nbsp    = counts["nbsp_survived"]
n_iframe  = counts["iframe"]
n_img     = counts["img"]
n_url     = counts["url_in_text"]
n_uni     = counts["high_unicode"]

print()
print(f"  Format:          {'✅ Uniform escaped-HTML' if ok else '⚠️  Mixed (escaped + raw HTML)'}")
print(f"  Completeness:    {'✅ All present' if total_bad == 0 else '❌ ' + str(total_bad) + ' missing'}")
print(f"  Control chars:   {'✅ None' if n_ctrl == 0 else '⚠️  ' + str(n_ctrl) + ' jobs'}")
print(f"  Mojibake:        {'✅ None' if n_moji == 0 else '⚠️  ' + str(n_moji) + ' jobs'}")
print(f"  &nbsp; leakage:  {'✅ None' if n_nbsp == 0 else '⚠️  ' + str(n_nbsp) + ' jobs after decoding'}")
print(f"  Iframes:         {'✅ None' if n_iframe == 0 else '⚠️  ' + str(n_iframe) + ' jobs (noise)'}")
print(f"  Images:          {'ℹ️  ' + str(n_img) + ' jobs (strip during parsing)' if n_img else '✅ None'}")
print(f"  Bare URLs:       {'ℹ️  ' + str(n_url) + ' jobs (decide: keep or strip)' if n_url else '✅ None'}")
print(f"  High-unicode:    {'ℹ️  ' + str(n_uni) + ' jobs (curly quotes, em-dash — likely intentional)' if n_uni else '✅ None'}")

print()
print("  Preprocessing pipeline should:")
print("    1. html.unescape() twice  (double-encoded entities → raw HTML)")
print("    2. BeautifulSoup parse    (strip all tags, drop iframes/imgs)")
print("    3. Normalise whitespace   (\\xa0 → space, collapse runs)")
print("    4. Optionally strip/keep  bare URLs depending on downstream use")
print()

conn.close()
