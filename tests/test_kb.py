import pytest

from mindbridge.kb import KBError, build_chunks, chunk_doc, load_docs, parse_doc

from .conftest import ROOT

KB = ROOT / "data" / "kb"


def test_every_document_loads_and_has_metadata():
    docs = load_docs(KB)
    assert len(docs) >= 25
    assert len({d.id for d in docs}) == len(docs)
    for d in docs:
        assert d.title and d.topics, d.id
        assert len(d.body.split()) >= 80, d.id


def test_chunks_are_small_and_traceable():
    chunks = build_chunks(KB)
    assert len({c.id for c in chunks}) == len(chunks)
    for c in chunks:
        assert c.text.strip()
        assert len(c.text.split()) <= 140
        assert c.id.startswith(c.doc_id + "#")


def test_long_section_is_split_on_paragraph_boundaries(tmp_path):
    para = " ".join(["word"] * 70)
    f = tmp_path / "long.md"
    f.write_text(f"---\ntitle: Long\ntopics: a\n---\n## Part\n{para}\n\n{para}\n\n{para}\n")
    chunks = chunk_doc(parse_doc(f), max_words=100)
    assert len(chunks) == 3
    assert all(c.section == "Part" for c in chunks)


def test_text_before_first_heading_becomes_overview(tmp_path):
    f = tmp_path / "intro.md"
    f.write_text("---\ntitle: T\ntopics: a\n---\nIntro text here.\n\n## Next\nMore.\n")
    chunks = chunk_doc(parse_doc(f))
    assert [c.section for c in chunks] == ["Overview", "Next"]


@pytest.mark.parametrize(
    "content",
    ["no front matter", "---\ntopics: a\n---\nbody", "---\ntitle: x\nbroken line\n---\nbody"],
)
def test_malformed_documents_are_rejected(tmp_path, content):
    f = tmp_path / "bad.md"
    f.write_text(content)
    with pytest.raises(KBError):
        parse_doc(f)


def test_empty_directory_is_an_error(tmp_path):
    with pytest.raises(KBError):
        load_docs(tmp_path)
