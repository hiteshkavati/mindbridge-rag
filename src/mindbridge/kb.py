from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


class KBError(ValueError):
    pass


@dataclass(frozen=True)
class Doc:
    id: str
    title: str
    topics: tuple[str, ...]
    body: str
    keywords: tuple[str, ...] = ()


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    title: str
    topics: tuple[str, ...]
    section: str
    text: str
    keywords: tuple[str, ...] = ()

    @property
    def index_text(self) -> str:
        # keywords are the phrases people actually type, which the prose does not always use
        extra = ", ".join((*self.topics, *self.keywords))
        return f"{self.title}. {extra}. {self.section}. {self.text}"


_FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n(.*)\Z", re.S)
_HEADING = re.compile(r"^## +(.+)$", re.M)


def _csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def parse_doc(path: Path) -> Doc:
    match = _FRONT_MATTER.match(path.read_text(encoding="utf-8"))
    if not match:
        raise KBError(f"{path.name}: missing front matter block")
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if not line.strip():
            continue
        key, sep, value = line.partition(":")
        if not sep:
            raise KBError(f"{path.name}: bad front matter line {line!r}")
        meta[key.strip()] = value.strip()
    if not meta.get("title"):
        raise KBError(f"{path.name}: title is required")
    topics = _csv(meta.get("topics", ""))
    keywords = _csv(meta.get("keywords", ""))
    return Doc(
        id=path.stem,
        title=meta["title"],
        topics=topics,
        body=match.group(2).strip(),
        keywords=keywords,
    )


def load_docs(kb_dir: Path | str) -> list[Doc]:
    files = sorted(Path(kb_dir).glob("*.md"))
    if not files:
        raise KBError(f"no markdown documents found in {kb_dir}")
    return [parse_doc(f) for f in files]


def _pack(paragraphs: list[str], max_words: int) -> list[str]:
    pieces: list[str] = []
    current: list[str] = []
    count = 0
    for para in paragraphs:
        n = len(para.split())
        if current and count + n > max_words:
            pieces.append("\n\n".join(current))
            current, count = [], 0
        current.append(para)
        count += n
    if current:
        pieces.append("\n\n".join(current))
    return pieces


def chunk_doc(doc: Doc, max_words: int = 110) -> list[Chunk]:
    """Split on '##' headings, then pack paragraphs inside a section up to max_words."""
    parts = _HEADING.split(doc.body)
    sections: list[tuple[str, str]] = []
    if parts[0].strip():
        sections.append(("Overview", parts[0]))
    for heading, text in zip(parts[1::2], parts[2::2], strict=True):
        sections.append((heading.strip(), text))

    chunks: list[Chunk] = []
    for heading, text in sections:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        for piece in _pack(paragraphs, max_words):
            chunks.append(
                Chunk(
                    id=f"{doc.id}#{len(chunks)}",
                    doc_id=doc.id,
                    title=doc.title,
                    topics=doc.topics,
                    section=heading,
                    text=piece,
                    keywords=doc.keywords,
                )
            )
    return chunks


def build_chunks(kb_dir: Path | str, max_words: int = 110) -> list[Chunk]:
    docs = load_docs(kb_dir)
    ids = [d.id for d in docs]
    if len(set(ids)) != len(ids):
        raise KBError("duplicate document ids")
    return [c for d in docs for c in chunk_doc(d, max_words)]
