"""Wikipedia-backed corpus for the RAG agent demo.

Fetches a fixed set of Wikipedia articles via the MediaWiki API, chunks them
into ~300-word passages, and caches the result to disk so subsequent runs are
offline-fast. The article titles span several domains (space, biology, history
of science, geography) so the corpus exercises a real retrieval problem rather
than a toy one.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import requests

CACHE_PATH = Path(__file__).parent / "wikipedia_corpus.json"

# A deliberately mixed set of Wikipedia titles. Includes the topics our example
# questions reference plus enough surrounding material that retrieval is non-
# trivial (i.e. the "right" passage isn't the only one with overlapping words).
TOPICS: List[str] = [
    # Space / astronomy
    "Voyager 1",
    "Voyager 2",
    "Pioneer 10",
    "Cassini-Huygens",
    "Hubble Space Telescope",
    "James Webb Space Telescope",
    "Heliopause",
    "Interstellar medium",
    # Biology / zoology
    "Tardigrade",
    "Okapi",
    "Aye-aye",
    "Giraffe",
    "Madagascar",
    # Earth science
    "Mid-Atlantic Ridge",
    "Marie Tharp",
    "Plate tectonics",
    "Burgess Shale",
    "Cambrian explosion",
    # History of computing & science
    "Ada Lovelace",
    "Charles Babbage",
    "Analytical Engine",
    "Antikythera mechanism",
    "Alan Turing",
    # General
    "William Shakespeare",
    "Hamlet",
    "Renaissance",
]

CHUNK_WORDS_TARGET = 300
CHUNK_WORDS_MIN = 80  # drop trailing tiny stubs

WIKI_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "uqlm-rag-demo/0.1 (https://github.com/cvs-health/uqlm)"


@dataclass
class Passage:
    id: str
    title: str
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


def _split_into_paragraphs(text: str) -> List[str]:
    # Strip section headings like "== History ==" and split on blank lines.
    text = re.sub(r"^=+.*?=+\s*$", "", text, flags=re.MULTILINE)
    parts = re.split(r"\n{2,}", text)
    return [p.strip() for p in parts if p.strip()]


def _chunk(paragraphs: List[str]) -> List[str]:
    chunks: List[str] = []
    buf: List[str] = []
    buf_words = 0
    for p in paragraphs:
        n = len(p.split())
        if buf_words + n > CHUNK_WORDS_TARGET and buf:
            chunks.append(" ".join(buf))
            buf, buf_words = [], 0
        buf.append(p)
        buf_words += n
    if buf and buf_words >= CHUNK_WORDS_MIN:
        chunks.append(" ".join(buf))
    return chunks


def _fetch_extract(title: str, session: requests.Session) -> Optional[str]:
    """Fetch the plain-text extract of a Wikipedia article."""
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": 1,
        "redirects": 1,
        "formatversion": 2,
    }
    r = session.get(WIKI_API, params=params, timeout=30)
    r.raise_for_status()
    pages = r.json().get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing"):
        return None
    return pages[0].get("extract")


def fetch_corpus(refresh: bool = False, polite_delay: float = 0.2) -> List[dict]:
    """Build (or load) the Wikipedia corpus. Returns a list of passage dicts.

    The corpus is cached at ``wikipedia_corpus.json`` next to this file.
    Pass ``refresh=True`` to re-download from Wikipedia.
    """
    if CACHE_PATH.exists() and not refresh:
        return json.loads(CACHE_PATH.read_text())

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    passages: List[Passage] = []
    for title in TOPICS:
        try:
            extract = _fetch_extract(title, session)
        except Exception as e:
            print(f"  skip {title!r}: {type(e).__name__}: {e}")
            continue
        if not extract:
            print(f"  skip {title!r}: no extract returned")
            continue
        chunks = _chunk(_split_into_paragraphs(extract))
        for i, text in enumerate(chunks):
            passages.append(
                Passage(id=f"{title}#{i}", title=title, text=text)
            )
        print(f"  {title}: {len(chunks)} passages")
        time.sleep(polite_delay)  # be a good API citizen

    data = [p.to_dict() for p in passages]
    CACHE_PATH.write_text(json.dumps(data, indent=1))
    print(f"\nWrote {len(data)} passages to {CACHE_PATH.name}")
    return data


def load_corpus(refresh: bool = False) -> List[dict]:
    """Convenience alias of fetch_corpus()."""
    return fetch_corpus(refresh=refresh)


if __name__ == "__main__":
    import sys

    refresh = "--refresh" in sys.argv
    docs = load_corpus(refresh=refresh)
    print(f"\nTotal: {len(docs)} passages across {len(TOPICS)} topics")
    if docs:
        avg = sum(len(d["text"].split()) for d in docs) / len(docs)
        print(f"Avg length: {avg:.0f} words")
