"""
Preprocessing module for job descriptions.

Converts raw HTML-encoded job descriptions into clean plain text through a 5-step pipeline:
1. Unescape HTML entities (handling double-encoding)
2. Strip iframes and images
3. Extract plain text from HTML
4. Normalize whitespace
5. Normalize Unicode punctuation to ASCII equivalents
"""

import html
import logging
import re
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Unicode replacements for Step 5
_UNICODE_TABLE = str.maketrans({
    '\u2018': "'",   # left single quotation mark
    '\u2019': "'",   # right single quotation mark
    '\u201c': '"',   # left double quotation mark
    '\u201d': '"',   # right double quotation mark
    '\u2014': '-',   # em dash
    '\u2013': '-',   # en dash
    '\u2026': '...',  # ellipsis
})

# Regex for collapsing whitespace (Step 4)
_WHITESPACE_RE = re.compile(r'[\s\xa0]+')


def _step1_unescape(raw: str) -> str:
    """Step 1: Unescape HTML entities, handling double-encoding."""
    for _ in range(5):
        decoded = html.unescape(raw)
        if decoded == raw:
            break
        raw = decoded
    return raw


def _step2_strip_iframes_and_images(html_text: str) -> str:
    """Step 2: Remove iframe and img tags using BeautifulSoup."""
    soup = BeautifulSoup(html_text, "html.parser")
    for tag in soup.find_all(["iframe", "img"]):
        tag.decompose()
    return str(soup)


def _step3_extract_text(html_text: str) -> str:
    """Step 3: Extract plain text from HTML, using space as separator between blocks."""
    soup = BeautifulSoup(html_text, "html.parser")
    return soup.get_text(separator=" ")


def _step4_normalize_whitespace(text: str) -> str:
    """Step 4: Collapse all whitespace sequences to single space and strip."""
    return _WHITESPACE_RE.sub(' ', text).strip()


def _step5_normalize_unicode_punctuation(text: str) -> str:
    """Step 5: Replace curly quotes, em-dashes, and ellipses with ASCII equivalents."""
    return text.translate(_UNICODE_TABLE)


def preprocess_description(raw: str) -> str:
    """
    Apply all 5 preprocessing steps to a raw job description.

    Args:
        raw: Raw HTML-encoded job description (may be None or empty).

    Returns:
        Clean plain text description, or empty string if input is None/empty.
    """
    if not raw:
        return ''

    text = _step1_unescape(raw)
    text = _step2_strip_iframes_and_images(text)
    text = _step3_extract_text(text)
    text = _step4_normalize_whitespace(text)
    text = _step5_normalize_unicode_punctuation(text)
    return text