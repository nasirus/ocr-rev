"""
Synthetic training data generator.

Renders synthetic text lines (letters/digits/whitespace/special chars)
onto images using Pillow, producing (image, label) pairs for CTC
training. No external labeled datasets needed — infinite free data.

Enhanced to produce realistic document-like text patterns including:
- Addresses, emails, URLs, invoice/reference numbers
- Currency amounts, phone numbers, dates with separators
- Proper English sentences and fragments
- Mixed-case identifiers and file paths
"""

from __future__ import annotations

import os
import string
import sys
from pathlib import Path

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
except ImportError:
    print(
        "Pillow is required for synthetic data generation.\n"
        "Install with: pip install pillow",
        file=sys.stderr,
    )
    raise

from microocr.model import CHARS
from microocr.preprocess import TARGET_HEIGHT, preprocess
from training.augment import augment

_ALNUM_CHARS = [c for c in CHARS if c.isalnum()]
_SPACE_CHAR = " "
_SPECIAL_CHARS = [c for c in CHARS if (not c.isalnum()) and c != _SPACE_CHAR]

# Rare characters that need frequency boosting
_RARE_CHARS = list("QZXqzxJKjk@#%")
# Common character pool (for mixing)
_COMMON_CHARS = list(CHARS)

# Pattern generators for diverse text
_CAMEL_WORDS = [
    "getData",
    "setName",
    "isValid",
    "hasError",
    "toString",
    "valueOf",
    "parseInt",
    "getItem",
    "onClick",
    "forEach",
    "indexOf",
    "toFixed",
    "charAt",
    "endsWith",
    "replace",
    "concat",
    "isEmpty",
    "toArray",
    "readFile",
    "sendMsg",
    "runTest",
    "logInfo",
    "mapKeys",
    "zipWith",
]

# Expanded phrase words to cover realistic document vocabulary
_PHRASE_WORDS = [
    "hello",
    "world",
    "micro",
    "ocr",
    "python",
    "token",
    "model",
    "image",
    "text",
    "decode",
    "encode",
    "framework",
    "fast",
    "tiny",
    "simple",
    "sample",
    "phone",
    "browser",
    "vision",
    "line",
    "space",
    "base64",
    "content",
    "batch",
    "train",
    "valid",
    "bench",
    # Document-related words
    "invoice",
    "receipt",
    "total",
    "amount",
    "payment",
    "order",
    "shipping",
    "address",
    "customer",
    "account",
    "number",
    "date",
    "price",
    "quantity",
    "subtotal",
    "discount",
    "balance",
    "due",
    "reference",
    "description",
    "item",
    "product",
    "service",
    "tax",
    "company",
    "report",
    "page",
    "document",
    "section",
    "table",
    "name",
    "email",
    "phone",
    "street",
    "city",
    "state",
    "country",
    "notes",
    "comments",
    "status",
    "pending",
    "approved",
    "complete",
    "error",
    "warning",
    "success",
    "failed",
    "active",
    "expired",
]

_SEPARATORS = [" ", "-", "_", "/", "."]
_TAIL_PUNCT = ["", "!", "?", ":", ";", "."]
_CONNECTOR_WORDS = [
    "and",
    "or",
    "but",
    "if",
    "when",
    "while",
    "because",
    "for",
    "with",
    "without",
    "from",
    "into",
    "over",
    "under",
    "to",
    "of",
    "in",
    "on",
    "by",
    "as",
    "that",
    "we",
    "you",
    "they",
    "it",
    "lets",
    "dont",
]

# ── Realistic document data pools ──────────────────────────────────────
_FIRST_NAMES = [
    "James",
    "Mary",
    "John",
    "Patricia",
    "Robert",
    "Jennifer",
    "Michael",
    "Linda",
    "David",
    "Sarah",
    "William",
    "Karen",
    "Richard",
    "Nancy",
    "Thomas",
    "Lisa",
    "Charles",
    "Betty",
    "Daniel",
    "Helen",
    "Alex",
    "Chris",
    "Sam",
    "Jordan",
    "Taylor",
    "Morgan",
    "Casey",
    "Robin",
]
_LAST_NAMES = [
    "Smith",
    "Johnson",
    "Williams",
    "Brown",
    "Jones",
    "Garcia",
    "Miller",
    "Davis",
    "Rodriguez",
    "Martinez",
    "Anderson",
    "Taylor",
    "Thomas",
    "Moore",
    "Jackson",
    "Martin",
    "Lee",
    "Thompson",
    "White",
    "Harris",
    "Clark",
    "Lewis",
    "Robinson",
    "Walker",
    "Young",
    "Allen",
    "King",
]
_STREET_SUFFIXES = [
    "St",
    "Ave",
    "Blvd",
    "Dr",
    "Ln",
    "Rd",
    "Way",
    "Ct",
    "Pl",
    "Cir",
]
_STREET_NAMES = [
    "Main",
    "Oak",
    "Pine",
    "Maple",
    "Cedar",
    "Elm",
    "Park",
    "Lake",
    "Hill",
    "River",
    "Spring",
    "Forest",
    "Sunset",
    "Valley",
    "Church",
    "Mill",
    "Market",
    "Center",
    "Broadway",
    "Washington",
    "Franklin",
]
_CITIES = [
    "New York",
    "Los Angeles",
    "Chicago",
    "Houston",
    "Phoenix",
    "Philadelphia",
    "San Antonio",
    "San Diego",
    "Dallas",
    "Austin",
    "Portland",
    "Denver",
    "Seattle",
    "Boston",
    "Atlanta",
    "Miami",
    "Detroit",
    "Minneapolis",
    "Cleveland",
    "Pittsburgh",
    "Oakland",
]
_STATES = [
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
]
_DOMAINS = [
    "gmail.com",
    "yahoo.com",
    "outlook.com",
    "company.com",
    "example.org",
    "mail.net",
    "work.io",
    "corp.co",
    "test.dev",
    "acme.com",
]
_URL_PATHS = [
    "docs",
    "api",
    "help",
    "about",
    "products",
    "services",
    "contact",
    "blog",
    "news",
    "support",
    "login",
    "dashboard",
    "settings",
    "reports",
    "files",
    "images",
    "download",
    "upload",
    "search",
]
_FILE_EXTENSIONS = [
    ".txt",
    ".pdf",
    ".csv",
    ".xlsx",
    ".docx",
    ".png",
    ".jpg",
    ".json",
    ".xml",
    ".html",
    ".py",
    ".js",
    ".ts",
    ".log",
]
_UNITS = ["kg", "lb", "oz", "ml", "L", "cm", "mm", "in", "ft", "m"]
_CURRENCY_SYMBOLS = ["$", "#"]  # using # as substitute for non-CHARS currencies

# Sentence templates — {N} = noun, {V} = verb, {A} = adjective
_NOUNS = [
    "file",
    "report",
    "order",
    "system",
    "user",
    "data",
    "page",
    "item",
    "result",
    "task",
    "code",
    "test",
    "error",
    "list",
    "table",
    "value",
    "input",
    "output",
    "query",
    "record",
    "request",
    "response",
    "server",
    "client",
    "process",
]
_VERBS = [
    "is",
    "was",
    "has",
    "will",
    "can",
    "should",
    "must",
    "may",
    "contains",
    "requires",
    "includes",
    "shows",
    "returns",
    "creates",
    "updates",
    "deletes",
    "reads",
    "writes",
    "sends",
    "loads",
    "saves",
    "runs",
    "starts",
    "stops",
    "fails",
]
_ADJECTIVES = [
    "new",
    "old",
    "valid",
    "invalid",
    "active",
    "pending",
    "open",
    "closed",
    "empty",
    "full",
    "large",
    "small",
    "first",
    "last",
    "next",
    "current",
    "total",
    "final",
    "primary",
    "default",
]
_ADVERBS = [
    "not",
    "also",
    "now",
    "then",
    "here",
    "only",
    "just",
    "still",
    "already",
    "never",
    "always",
    "often",
    "very",
    "well",
    "soon",
]

# Key-value label patterns found in forms/documents
# Words with commonly confused character pairs for targeted training
_CONFUSION_WORDS = [
    # p/d/b/q heavy
    "deploy",
    "depend",
    "adapt",
    "display",
    "upload",
    "develop",
    "rapid",
    "deep",
    "pad",
    "dip",
    "drop",
    "drape",
    "typed",
    "python",
    "php",
    "pandas",
    "dedup",
    "paddle",
    "stopped",
    "bumped",
    "probed",
    "grabbed",
    "popped",
    "mapped",
    "dripped",
    "wrapped",
    "deposit",
    "republic",
    "backdrop",
    "doorstep",
    "endpoint",
    "dropbox",
    "blueprint",
    # g/c/q heavy
    "graphic",
    "magic",
    "garage",
    "logic",
    "organic",
    "gigantic",
    "cargo",
    "recognize",
    "glycogen",
    "changing",
    "charging",
    "engaging",
    "packaging",
    "configuring",
    "debugging",
    "queuing",
    "conquering",
    "collaging",
    # l/i/1 heavy
    "illegal",
    "illicit",
    "initial",
    "install",
    "literal",
    "inline",
    "pill",
    "fill",
    "bill",
    "still",
    "skill",
    "llm",
    "dll",
    "all",
    "tall",
    "wall",
    "hall",
    "illustrate",
    "illumination",
    "illusion",
    "liability",
    "lollipop",
    "parallel",
    "milliliter",
    "vanilla",
    # j/i heavy
    "jinja",
    "jit",
    "fiji",
    "hijack",
    "jigsaw",
    "majin",
    # y/v heavy
    "every",
    "heavy",
    "survey",
    "valley",
    "voyage",
    "very",
    "ivy",
    "savvy",
    "levy",
    "navy",
    "wavy",
    "victory",
    "variety",
    # double letters (CTC challenge)
    "balloon",
    "coffee",
    "committee",
    "assess",
    "parallel",
    "broccoli",
    "accommodate",
    "announce",
    "approve",
    "arrange",
    "arrived",
    "occurred",
    "succeed",
    "suppress",
    "tomorrow",
    "possess",
    "misspell",
    "millennium",
    "bookkeeper",
    "successfully",
    # comma/period disambiguation
    "e.g.",
    "i.e.",
    "Dr.",
    "Mr.",
    "Ms.",
    "etc.",
    "vs.",
    "approx.",
    # apostrophe-containing words (CTC + punctuation challenge)
    "let's",
    "don't",
    "it's",
    "can't",
    "won't",
    "I'm",
    "they're",
    "we're",
    "you're",
    "he's",
    "she's",
    "that's",
    "there's",
    "what's",
]

_FIELD_LABELS = [
    "Name",
    "Date",
    "Time",
    "Total",
    "Amount",
    "Price",
    "Qty",
    "Tax",
    "Ref",
    "ID",
    "No",
    "Page",
    "Item",
    "Code",
    "Type",
    "Status",
    "Phone",
    "Email",
    "Fax",
    "Note",
    "Memo",
    "From",
    "To",
    "CC",
    "Subject",
    "Dept",
    "Div",
    "Acct",
    "PO",
    "SO",
]


def _generate_text(rng: np.random.Generator, min_len: int, max_len: int) -> str:
    """Generate diverse text patterns reflecting real documents.

    Produces a mix of:
    - Pure random alphanumeric strings
    - CamelCase developer words
    - Alphanumeric codes (e.g., AB12cd)
    - Date-like strings with various separators
    - Multi-token text with spaces and separators
    - Strings with boosted rare characters
    - Email addresses
    - URLs and file paths
    - Addresses (street, city/state/zip)
    - Phone numbers
    - Currency amounts
    - Invoice/reference numbers
    - Natural English sentence fragments
    - Key-value pairs (label: value)
    - Measurement values with units
    """
    choice = rng.random()

    if choice < 0.08:
        # Standard random alphanumeric
        length = int(rng.integers(min_len, max_len + 1))
        text = "".join(str(rng.choice(_ALNUM_CHARS)) for _ in range(length))
        return _fit_text_length(text, rng, min_len, max_len)

    if choice < 0.13:
        # CamelCase words
        word = str(rng.choice(_CAMEL_WORDS))
        if rng.random() < 0.3:
            word = word + str(int(rng.integers(0, 100)))
        return _fit_text_length(word, rng, min_len, max_len)

    if choice < 0.18:
        # Alphanumeric codes (e.g., AB12cd, X7y9Z)
        length = int(rng.integers(max(min_len, 4), min(max_len + 1, 16)))
        code = []
        for _ in range(length):
            pool_choice = rng.random()
            if pool_choice < 0.35:
                code.append(str(rng.choice(list(string.ascii_uppercase))))
            elif pool_choice < 0.65:
                code.append(str(rng.choice(list(string.digits))))
            else:
                code.append(str(rng.choice(list(string.ascii_lowercase))))
        return _fit_text_length("".join(code), rng, min_len, max_len)

    if choice < 0.23:
        return _fit_text_length(_gen_date(rng, max_len), rng, min_len, max_len)
    if choice < 0.28:
        return _fit_text_length(_gen_email(rng, max_len), rng, min_len, max_len)
    if choice < 0.32:
        return _fit_text_length(_gen_url_or_path(rng, max_len), rng, min_len, max_len)
    if choice < 0.36:
        return _fit_text_length(_gen_address(rng, max_len), rng, min_len, max_len)
    if choice < 0.40:
        return _fit_text_length(_gen_phone(rng, max_len), rng, min_len, max_len)
    if choice < 0.46:
        return _fit_text_length(_gen_currency(rng, max_len), rng, min_len, max_len)
    if choice < 0.51:
        return _fit_text_length(
            _gen_reference_number(rng, max_len), rng, min_len, max_len
        )
    if choice < 0.60:
        return _fit_text_length(_gen_sentence(rng, max_len), rng, min_len, max_len)
    if choice < 0.67:
        return _fit_text_length(_gen_key_value(rng, max_len), rng, min_len, max_len)
    if choice < 0.72:
        return _fit_text_length(_gen_measurement(rng, max_len), rng, min_len, max_len)
    if choice < 0.78:
        return _fit_text_length(_gen_full_name(rng, max_len), rng, min_len, max_len)
    if choice < 0.82:
        # Paragraph-like fragments to better match long-form OCR inputs.
        return _gen_paragraph_fragment(rng, min_len=min_len, max_len=max_len)
    if choice < 0.96:
        # Confusion-pair enriched text for commonly confused characters
        return _gen_confusion_pair_text(rng, min_len=min_len, max_len=max_len)

    # Rare/special boosted strings
    length = int(rng.integers(min_len, max_len + 1))
    text = []
    for _ in range(length):
        p = rng.random()
        if p < 0.35:
            text.append(str(rng.choice(_RARE_CHARS)))
        elif p < 0.65 and _SPECIAL_CHARS:
            text.append(str(rng.choice(_SPECIAL_CHARS)))
        elif p < 0.75:
            text.append(_SPACE_CHAR)
        else:
            text.append(str(rng.choice(_COMMON_CHARS)))
    out = "".join(text).strip()
    return _fit_text_length(out or "0", rng, min_len, max_len)


def _fit_text_length(
    text: str,
    rng: np.random.Generator,
    min_len: int,
    max_len: int,
) -> str:
    """Normalize generated text to requested length bounds."""
    filtered = "".join(c for c in text if c in CHARS)
    filtered = " ".join(filtered.split())
    if not filtered:
        filtered = str(rng.choice(_PHRASE_WORDS))

    if len(filtered) > max_len:
        filtered = filtered[:max_len].rstrip(" ,;:-_/")
    while len(filtered) < min_len:
        token = str(rng.choice(_PHRASE_WORDS))
        if len(filtered) + 1 + len(token) <= max_len:
            filtered = f"{filtered} {token}" if filtered else token
        else:
            break
    return filtered[:max_len].strip() or "text"


def _gen_paragraph_fragment(
    rng: np.random.Generator,
    min_len: int,
    max_len: int,
) -> str:
    """Generate paragraph-like text with mostly natural spacing/casing."""
    low = min(max_len, max(min_len, 14))
    if max_len >= 96:
        low = max(low, int(max_len * 0.55))
    elif max_len >= 72:
        low = max(low, int(max_len * 0.45))
    high = max_len + 1
    if high <= low:
        high = low + 1
    target = int(rng.integers(low, high))
    parts: list[str] = []

    while len(" ".join(parts)) < target:
        p = rng.random()
        if p < 0.68:
            token = str(rng.choice(_PHRASE_WORDS))
        elif p < 0.88:
            token = str(rng.choice(_CONNECTOR_WORDS))
        else:
            token = str(rng.choice(_CAMEL_WORDS)).lower()

        if rng.random() < 0.06:
            token = token + str(int(rng.integers(0, 100)))
        parts.append(token)

    if len(parts) > 5 and rng.random() < 0.55:
        comma_idx = int(rng.integers(2, len(parts) - 1))
        parts[comma_idx] = parts[comma_idx] + ","

    text = " ".join(parts)
    if rng.random() < 0.45:
        text = text + "."
    elif rng.random() < 0.12:
        text = text + "?"

    if rng.random() < 0.35 and text:
        text = text[0].upper() + text[1:]
    return _fit_text_length(text, rng, min_len, max_len)


# ── Document-style text generators ────────────────────────────────────


def _gen_date(rng: np.random.Generator, max_len: int) -> str:
    """Generate date strings in various realistic formats."""
    mm = int(rng.integers(1, 13))
    dd = int(rng.integers(1, 29))
    yyyy = int(rng.integers(1990, 2030))
    yy = yyyy % 100

    style = int(rng.integers(0, 8))
    if style == 0:
        text = f"{mm:02d}/{dd:02d}/{yyyy:04d}"  # 01/15/2024
    elif style == 1:
        text = f"{yyyy:04d}-{mm:02d}-{dd:02d}"  # 2024-01-15
    elif style == 2:
        text = f"{dd:02d}.{mm:02d}.{yyyy:04d}"  # 15.01.2024
    elif style == 3:
        text = f"{mm:02d}-{dd:02d}-{yy:02d}"  # 01-15-24
    elif style == 4:
        text = f"{mm:02d}{dd:02d}{yyyy:04d}"  # 01152024
    elif style == 5:
        text = f"{yyyy:04d}{mm:02d}{dd:02d}"  # 20240115
    elif style == 6:
        text = f"{dd:02d}/{mm:02d}/{yy:02d}"  # 15/01/24
    else:
        text = f"{dd:02d}{mm:02d}{yy:02d}"  # 150124
    return text[:max_len]


def _gen_email(rng: np.random.Generator, max_len: int) -> str:
    """Generate realistic email addresses."""
    first = str(rng.choice(_FIRST_NAMES)).lower()
    last = str(rng.choice(_LAST_NAMES)).lower()
    domain = str(rng.choice(_DOMAINS))

    style = int(rng.integers(0, 5))
    if style == 0:
        local = f"{first}.{last}"
    elif style == 1:
        local = f"{first}{last}"
    elif style == 2:
        local = f"{first[0]}{last}"
    elif style == 3:
        local = f"{first}_{last}{int(rng.integers(1, 100))}"
    else:
        local = f"{first}{int(rng.integers(10, 999))}"

    # Filter to only CHARS-safe characters
    email = f"{local}@{domain}"
    email = "".join(c for c in email if c in CHARS)
    return email[:max_len]


def _gen_url_or_path(rng: np.random.Generator, max_len: int) -> str:
    """Generate URLs or file paths."""
    if rng.random() < 0.5:
        # URL-like
        domain = str(rng.choice(_DOMAINS))
        path = str(rng.choice(_URL_PATHS))
        if rng.random() < 0.3:
            text = f"www.{domain}/{path}"
        else:
            text = f"{domain}/{path}"
        if rng.random() < 0.3:
            text += f"/{str(rng.choice(_URL_PATHS))}"
    else:
        # File path
        parts = [str(rng.choice(_URL_PATHS)) for _ in range(int(rng.integers(2, 5)))]
        ext = str(rng.choice(_FILE_EXTENSIONS))
        filename = str(rng.choice(_PHRASE_WORDS)) + ext
        text = "/".join(parts) + "/" + filename

    text = "".join(c for c in text if c in CHARS)
    return text[:max_len]


def _gen_address(rng: np.random.Generator, max_len: int) -> str:
    """Generate street address components."""
    num = int(rng.integers(1, 9999))
    street = str(rng.choice(_STREET_NAMES))
    suffix = str(rng.choice(_STREET_SUFFIXES))
    style = int(rng.integers(0, 5))

    if style == 0:
        # Full street address
        text = f"{num} {street} {suffix}"
    elif style == 1:
        # City, State ZIP
        city = str(rng.choice(_CITIES))
        state = str(rng.choice(_STATES))
        zipcode = int(rng.integers(10000, 99999))
        text = f"{city}, {state} {zipcode}"
    elif style == 2:
        # Street + apt/unit
        apt = int(rng.integers(1, 999))
        if rng.random() < 0.5:
            text = f"{num} {street} {suffix} #{apt}"
        else:
            text = f"{num} {street} {suffix}, Apt {apt}"
    else:
        # Just ZIP or state
        zipcode = int(rng.integers(10000, 99999))
        state = str(rng.choice(_STATES))
        text = f"{state} {zipcode}"

    text = "".join(c for c in text if c in CHARS)
    return text[:max_len]


def _gen_phone(rng: np.random.Generator, max_len: int) -> str:
    """Generate phone number strings in various formats."""
    area = int(rng.integers(200, 999))
    mid = int(rng.integers(200, 999))
    last = int(rng.integers(1000, 9999))

    style = int(rng.integers(0, 5))
    if style == 0:
        text = f"({area}) {mid}-{last}"
    elif style == 1:
        text = f"{area}-{mid}-{last}"
    elif style == 2:
        text = f"{area}.{mid}.{last}"
    elif style == 3:
        text = f"+1 {area} {mid} {last}"
    else:
        text = f"{area}{mid}{last}"

    text = "".join(c for c in text if c in CHARS)
    return text[:max_len]


def _gen_currency(rng: np.random.Generator, max_len: int) -> str:
    """Generate currency amount strings."""
    symbol = str(rng.choice(_CURRENCY_SYMBOLS))
    style = int(rng.integers(0, 5))

    if style == 0:
        # Standard price
        dollars = int(rng.integers(1, 9999))
        cents = int(rng.integers(0, 100))
        text = f"{symbol}{dollars}.{cents:02d}"
    elif style == 1:
        # Large amount with commas simulated by dots/spaces
        amount = int(rng.integers(1000, 999999))
        text = f"{symbol}{amount:,}".replace(",", ",")
        cents = int(rng.integers(0, 100))
        text += f".{cents:02d}"
    elif style == 2:
        # Simple integer
        amount = int(rng.integers(1, 99999))
        text = f"{symbol}{amount}"
    elif style == 3:
        # With label
        dollars = int(rng.integers(1, 9999))
        cents = int(rng.integers(0, 100))
        label = str(rng.choice(["Total", "Amount", "Price", "Balance", "Due"]))
        text = f"{label}: {symbol}{dollars}.{cents:02d}"
    else:
        # Negative amount
        dollars = int(rng.integers(1, 999))
        cents = int(rng.integers(0, 100))
        text = f"-{symbol}{dollars}.{cents:02d}"

    text = "".join(c for c in text if c in CHARS)
    return text[:max_len]


def _gen_reference_number(rng: np.random.Generator, max_len: int) -> str:
    """Generate invoice/reference/order numbers."""
    prefix_options = [
        "INV",
        "REF",
        "ORD",
        "PO",
        "SO",
        "TXN",
        "ID",
        "DOC",
        "REC",
        "ACT",
        "JOB",
        "WO",
        "RMA",
        "SKU",
        "LOT",
        "SN",
    ]
    prefix = str(rng.choice(prefix_options))

    style = int(rng.integers(0, 5))
    if style == 0:
        num = int(rng.integers(1000, 999999))
        text = f"{prefix}-{num}"
    elif style == 1:
        num = int(rng.integers(100000, 9999999))
        text = f"{prefix}{num}"
    elif style == 2:
        # With date component
        yy = int(rng.integers(20, 27))
        seq = int(rng.integers(1, 9999))
        text = f"{prefix}-{yy}-{seq:04d}"
    elif style == 3:
        # Alphanumeric
        length = int(rng.integers(6, 12))
        chars = "".join(
            str(rng.choice(list(string.ascii_uppercase + string.digits)))
            for _ in range(length)
        )
        text = f"{prefix}-{chars}"
    else:
        num = int(rng.integers(1, 99999))
        text = f"#{num:05d}"

    return text[:max_len]


def _gen_sentence(rng: np.random.Generator, max_len: int) -> str:
    """Generate natural English sentence fragments."""
    style = int(rng.integers(0, 7))

    if style == 0:
        # Simple subject-verb pattern
        noun = str(rng.choice(_NOUNS))
        verb = str(rng.choice(_VERBS))
        adj = str(rng.choice(_ADJECTIVES))
        text = f"The {noun} {verb} {adj}."
    elif style == 1:
        # Action sentence
        noun = str(rng.choice(_NOUNS))
        verb = str(rng.choice(_VERBS))
        noun2 = str(rng.choice(_NOUNS))
        text = f"{noun} {verb} the {noun2}"
    elif style == 2:
        # With adverb
        noun = str(rng.choice(_NOUNS))
        verb = str(rng.choice(_VERBS))
        adv = str(rng.choice(_ADVERBS))
        text = f"The {noun} {adv} {verb}."
    elif style == 3:
        # Question
        noun = str(rng.choice(_NOUNS))
        verb = str(rng.choice(_VERBS))
        adj = str(rng.choice(_ADJECTIVES))
        text = f"Is the {noun} {adj}?"
    elif style == 4:
        # Imperative
        verb = str(rng.choice(_VERBS))
        noun = str(rng.choice(_NOUNS))
        text = f"Please {verb} the {noun}."
    elif style == 5:
        # Error/status message
        noun = str(rng.choice(_NOUNS))
        adj = str(rng.choice(_ADJECTIVES))
        text = f"Error: {noun} is {adj}"
    else:
        # Multi-clause
        noun1 = str(rng.choice(_NOUNS))
        verb1 = str(rng.choice(_VERBS))
        noun2 = str(rng.choice(_NOUNS))
        verb2 = str(rng.choice(_VERBS))
        text = f"The {noun1} {verb1} and {noun2} {verb2}"

    text = "".join(c for c in text if c in CHARS)
    if len(text) > max_len:
        text = text[:max_len].rstrip()
    return text or "test"


def _gen_key_value(rng: np.random.Generator, max_len: int) -> str:
    """Generate key-value pairs like form fields."""
    label = str(rng.choice(_FIELD_LABELS))
    style = int(rng.integers(0, 6))

    if style == 0:
        # Numeric value
        val = str(int(rng.integers(1, 99999)))
        text = f"{label}: {val}"
    elif style == 1:
        # Text value
        val = str(rng.choice(_PHRASE_WORDS))
        text = f"{label}: {val}"
    elif style == 2:
        # Date value
        mm = int(rng.integers(1, 13))
        dd = int(rng.integers(1, 29))
        yyyy = int(rng.integers(2020, 2027))
        text = f"{label}: {mm:02d}/{dd:02d}/{yyyy}"
    elif style == 3:
        # Boolean-ish
        val = str(rng.choice(["Yes", "No", "N/A", "TBD", "OK"]))
        text = f"{label}: {val}"
    elif style == 4:
        # With equals sign
        val = str(int(rng.integers(0, 9999)))
        text = f"{label} = {val}"
    else:
        # Bracketed
        val = str(rng.choice(_PHRASE_WORDS))
        text = f"{label} [{val}]"

    text = "".join(c for c in text if c in CHARS)
    return text[:max_len]


def _gen_measurement(rng: np.random.Generator, max_len: int) -> str:
    """Generate measurement values with units."""
    unit = str(rng.choice(_UNITS))
    style = int(rng.integers(0, 4))

    if style == 0:
        # Integer value
        val = int(rng.integers(1, 9999))
        text = f"{val} {unit}"
    elif style == 1:
        # Decimal value
        whole = int(rng.integers(0, 999))
        frac = int(rng.integers(0, 100))
        text = f"{whole}.{frac:02d} {unit}"
    elif style == 2:
        # Range
        v1 = int(rng.integers(1, 500))
        v2 = v1 + int(rng.integers(1, 500))
        text = f"{v1}-{v2} {unit}"
    else:
        # Dimension (e.g., 10 x 20 cm)
        v1 = int(rng.integers(1, 999))
        v2 = int(rng.integers(1, 999))
        text = f"{v1} x {v2} {unit}"

    text = "".join(c for c in text if c in CHARS)
    return text[:max_len]


def _gen_full_name(rng: np.random.Generator, max_len: int) -> str:
    """Generate full names in various formats."""
    first = str(rng.choice(_FIRST_NAMES))
    last = str(rng.choice(_LAST_NAMES))

    style = int(rng.integers(0, 5))
    if style == 0:
        text = f"{first} {last}"
    elif style == 1:
        text = f"{last}, {first}"
    elif style == 2:
        text = f"{first[0]}. {last}"
    elif style == 3:
        middle = str(rng.choice(_FIRST_NAMES))
        text = f"{first} {middle[0]}. {last}"
    else:
        title = str(rng.choice(["Mr", "Ms", "Dr"]))
        text = f"{title}. {first} {last}"

    text = "".join(c for c in text if c in CHARS)
    return text[:max_len]


def _gen_confusion_pair_text(
    rng: np.random.Generator, min_len: int, max_len: int
) -> str:
    """Generate text enriched with commonly confused character groups.

    Focuses on p/d/b/q, g/c/q, l/i/j, y/v, double letters, and punctuation
    to train the model on visually similar characters.
    """
    style = int(rng.integers(0, 4))

    if style == 0:
        # String of confusion words joined with spaces
        n_words = int(rng.integers(2, 8))
        words = [str(rng.choice(_CONFUSION_WORDS)) for _ in range(n_words)]
        text = " ".join(words)
    elif style == 1:
        # Mix confusion words with connectors for natural flow
        parts: list[str] = []
        n_words = int(rng.integers(3, 10))
        for i in range(n_words):
            if i > 0 and rng.random() < 0.35:
                parts.append(str(rng.choice(_CONNECTOR_WORDS)))
            parts.append(str(rng.choice(_CONFUSION_WORDS)))
        text = " ".join(parts)
        if rng.random() < 0.4:
            # Insert commas for punctuation training
            if len(parts) > 3:
                idx = int(rng.integers(2, len(parts) - 1))
                parts[idx] = parts[idx] + ","
            text = " ".join(parts)
    elif style == 2:
        # Confusion chars mixed with sentence fragments
        noun = str(rng.choice(_NOUNS))
        verb = str(rng.choice(_VERBS))
        cword1 = str(rng.choice(_CONFUSION_WORDS))
        cword2 = str(rng.choice(_CONFUSION_WORDS))
        templates = [
            f"the {noun} {verb} {cword1} and {cword2}",
            f"{cword1}, {cword2}: {noun} {verb}",
            f"please {verb} the {cword1} for {cword2}.",
            f"{cword1} is {verb} by {cword2}, not {noun}.",
        ]
        text = str(rng.choice(templates))
    elif style == 3:
        # Punctuation-heavy text with commas and periods
        words = [
            str(rng.choice(_CONFUSION_WORDS)) for _ in range(int(rng.integers(3, 7)))
        ]
        punct_templates = [
            f"{words[0]}, {words[1]}. {words[2]}",
            f"e.g., {words[0]}; i.e., {words[1]}",
            f"Dr. {str(rng.choice(_FIRST_NAMES))}, {words[0]} dept.",
            f"{words[0]}: {words[1]}, {words[2]}.",
        ]
        text = str(rng.choice(punct_templates))
    else:
        # Style 4: Comma vs period + apostrophe disambiguation
        apostrophe_words = [w for w in _CONFUSION_WORDS if "'" in w]
        if not apostrophe_words:
            apostrophe_words = ["let's", "don't", "it's"]
        cwords = [
            str(rng.choice(_CONFUSION_WORDS)) for _ in range(int(rng.integers(2, 5)))
        ]
        aword = str(rng.choice(apostrophe_words))
        punct_templates = [
            f"{aword}, {cwords[0]}. {cwords[1]}",
            f"{cwords[0]}, {aword}; {cwords[1]}.",
            f"i.e., {aword} and {cwords[0]}, not {cwords[1]}.",
            f"{aword}: {cwords[0]}, e.g., {cwords[1]}.",
            f"Note: {aword}, {cwords[0]}. See also {cwords[1]}.",
        ]
        text = str(rng.choice(punct_templates))

    text = "".join(c for c in text if c in CHARS)
    return _fit_text_length(text, rng, min_len, max_len)


def generate_sample(
    rng: np.random.Generator | None = None,
    min_len: int = 1,
    max_len: int = 20,
    font_size_range: tuple[int, int] = (20, 40),
    target_height: int = TARGET_HEIGHT,
    apply_augment: bool = True,
    align_with_inference: bool = True,
) -> tuple[np.ndarray, str]:
    """Generate a single synthetic (image, label) pair.

    Args:
        rng: Random generator for reproducibility.
        min_len: Minimum text length.
        max_len: Maximum text length.
        font_size_range: Range of font sizes to sample from.
        target_height: Target image height after preprocessing.
        apply_augment: Whether to apply data augmentation.
        align_with_inference: If True, run the same crop/resize pipeline
            as runtime inference.

    Returns:
        Tuple of:
            - 2-D float32 array of shape (target_height, W), values in [0, 1]
            - Label string
    """
    if rng is None:
        rng = np.random.default_rng()

    # Generate diverse text
    label = _generate_text(rng, min_len, max_len)

    # Keep long labels readable by biasing to smaller fonts for long strings.
    font_size = _sample_font_size_for_label(
        len(label),
        font_size_range=font_size_range,
        rng=rng,
    )

    # Render text to image (with optional variable kerning)
    use_variable_kerning = rng.random() < 0.3
    if use_variable_kerning:
        img = _render_text_variable_kerning(label, font_size, rng)
    else:
        img = _render_text(label, font_size, rng)

    # Augment
    img_f = img.astype(np.float32) / 255.0
    if apply_augment:
        img_f = augment(img_f, rng)

    img_u8 = np.clip(img_f * 255.0, 0.0, 255.0).astype(np.uint8)
    if align_with_inference:
        img_out = preprocess(
            img_u8,
            target_height=target_height,
            already_binary=False,
            resize_mode="bilinear",
        )
    else:
        from microocr.preprocess import resize_height

        img_out = (
            resize_height(img_u8, target_height, mode="bilinear").astype(np.float32)
            / 255.0
        )

    return img_out, label


def _sample_font_size_for_label(
    label_len: int,
    font_size_range: tuple[int, int],
    rng: np.random.Generator,
) -> int:
    """Sample a font size while controlling extreme line widths."""
    low, high = font_size_range
    if high <= low:
        high = low + 1

    # np.random.integers upper bound is exclusive.
    if label_len >= 110:
        high = min(high, 22)
    elif label_len >= 96:
        high = min(high, 24)
    elif label_len >= 80:
        high = min(high, 26)
    elif label_len >= 64:
        high = min(high, 28)
    elif label_len >= 48:
        high = min(high, 30)
    elif label_len >= 36:
        high = min(high, 33)

    if high <= low:
        low = max(8, high - 2)
    return int(rng.integers(low, high))


def generate_batch(
    batch_size: int,
    rng: np.random.Generator | None = None,
    **kwargs,
) -> tuple[list[np.ndarray], list[str]]:
    """Generate a batch of synthetic samples.

    Args:
        batch_size: Number of samples to generate.
        rng: Random generator.
        **kwargs: Passed to :func:`generate_sample`.

    Returns:
        Tuple of (list of images, list of labels).
    """
    if rng is None:
        rng = np.random.default_rng()

    images = []
    labels = []
    for _ in range(batch_size):
        img, label = generate_sample(rng=rng, **kwargs)
        images.append(img)
        labels.append(label)

    return images, labels


def _render_text(
    text: str,
    font_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Render text to a grayscale numpy array using Pillow.

    Supports variable ink darkness and background tones to simulate
    real scanned documents with faded ink, aged paper, etc.

    Args:
        text: The string to render.
        font_size: Font size in pixels.
        rng: Random generator (for font selection, padding, colors).

    Returns:
        2-D uint8 grayscale array.
    """
    font = _get_font(font_size, rng)

    # Create a temporary image to measure text size
    tmp = Image.new("L", (1, 1), color=255)
    draw = ImageDraw.Draw(tmp)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Add random padding
    pad_x = int(rng.integers(2, 14))
    pad_y = int(rng.integers(2, 10))
    img_w = text_w + 2 * pad_x
    img_h = text_h + 2 * pad_y

    # Variable background and ink colors for realism
    bg_color = int(rng.integers(230, 256))  # near-white background
    ink_color = int(rng.integers(0, 60))  # near-black ink

    # Occasionally simulate low-contrast text (faded ink / light print)
    if rng.random() < 0.15:
        ink_color = int(rng.integers(50, 120))  # faded ink
    if rng.random() < 0.10:
        bg_color = int(rng.integers(200, 230))  # aged/tinted paper

    # Render
    img = Image.new("L", (img_w, img_h), color=bg_color)
    draw = ImageDraw.Draw(img)
    draw.text((pad_x - bbox[0], pad_y - bbox[1]), text, fill=ink_color, font=font)

    return np.array(img, dtype=np.uint8)


def _render_text_variable_kerning(
    text: str,
    font_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Render text character-by-character with random inter-character spacing.

    This simulates variable kerning found in real-world text.
    Supports variable ink darkness and background tones.
    """
    font = _get_font(font_size, rng)
    tmp = Image.new("L", (1, 1), color=255)
    draw = ImageDraw.Draw(tmp)

    # Measure each character
    char_widths = []
    char_heights = []
    for ch in text:
        bbox = draw.textbbox((0, 0), ch, font=font)
        char_widths.append(bbox[2] - bbox[0])
        char_heights.append(bbox[3] - bbox[1])

    if not char_widths:
        return np.full((font_size + 4, 4), 255, dtype=np.uint8)

    max_h = max(char_heights) if char_heights else font_size

    # Compute total width with variable spacing
    spacings: list[int] = []
    for i in range(max(0, len(text) - 1)):
        # Avoid collapsing spaces visually: keep a clear gap around them.
        if text[i] == " " or text[i + 1] == " ":
            spacings.append(int(rng.integers(2, 9)))
        else:
            spacings.append(int(rng.integers(-1, 4)))
    total_w = sum(char_widths) + sum(spacings)

    pad_x = int(rng.integers(2, 14))
    pad_y = int(rng.integers(2, 10))
    img_w = max(total_w + 2 * pad_x, 4)
    img_h = max_h + 2 * pad_y

    # Variable background and ink colors
    bg_color = int(rng.integers(230, 256))
    ink_color = int(rng.integers(0, 60))
    if rng.random() < 0.15:
        ink_color = int(rng.integers(50, 120))
    if rng.random() < 0.10:
        bg_color = int(rng.integers(200, 230))

    img = Image.new("L", (img_w, img_h), color=bg_color)
    draw = ImageDraw.Draw(img)

    x_cursor = pad_x
    for i, ch in enumerate(text):
        bbox = draw.textbbox((0, 0), ch, font=font)
        # Slight per-character ink variation
        char_ink = max(0, min(255, ink_color + int(rng.integers(-10, 11))))
        draw.text((x_cursor - bbox[0], pad_y - bbox[1]), ch, fill=char_ink, font=font)
        x_cursor += char_widths[i]
        if i < len(spacings):
            x_cursor += spacings[i]

    return np.array(img, dtype=np.uint8)


# Cache discovered fonts
_font_cache: list[str] = []
_font_cache_by_category: dict[str, list[str]] = {}
_default_font_cache: ImageFont.FreeTypeFont | None = None


def _get_font(
    size: int, rng: np.random.Generator
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Get a random system font, or fall back to Pillow's default.

    Categorizes fonts when possible to allow weighted sampling of
    serif, sans-serif, and monospace families.
    """
    global _font_cache, _default_font_cache

    # Try to discover system fonts (first call only)
    if not _font_cache:
        _font_cache = _discover_fonts()

    if _font_cache:
        font_path = rng.choice(_font_cache)
        try:
            return ImageFont.truetype(font_path, size)
        except (OSError, IOError):
            pass

    # Fallback: Pillow's built-in default
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except (OSError, IOError):
        pass

    return ImageFont.load_default()


def _discover_fonts() -> list[str]:
    """Find TrueType fonts on the system."""
    font_dirs = [
        "/usr/share/fonts",
        "/usr/local/share/fonts",
        os.path.expanduser("~/.fonts"),
        os.path.expanduser("~/.local/share/fonts"),
        # macOS
        "/System/Library/Fonts",
        "/Library/Fonts",
        os.path.expanduser("~/Library/Fonts"),
        # Windows
        r"C:\Windows\Fonts",
    ]

    fonts: list[str] = []
    for d in font_dirs:
        p = Path(d)
        if p.is_dir():
            for f in p.rglob("*.ttf"):
                fonts.append(str(f))
            for f in p.rglob("*.TTF"):
                fonts.append(str(f))
            for f in p.rglob("*.otf"):
                fonts.append(str(f))

    return fonts
