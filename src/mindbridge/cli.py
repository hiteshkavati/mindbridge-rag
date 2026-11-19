from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import Settings, get_settings


def _cmd_serve(args: argparse.Namespace, cfg: Settings) -> int:
    import uvicorn

    from .api import create_app

    uvicorn.run(create_app(cfg), host=args.host, port=args.port, log_level="info")
    return 0


def _cmd_chat(args: argparse.Namespace, cfg: Settings) -> int:
    from .bootstrap import build_pipeline
    from .sessions import Turn

    pipeline = build_pipeline(cfg)
    print(
        f"MindBridge ({pipeline.llm.name}, {pipeline.retriever.embedder.name}). Ctrl-D to quit.\n"
    )
    history: list[Turn] = []
    while True:
        try:
            message = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not message:
            continue
        reply = pipeline.answer(message, history)
        history += [Turn("user", message), Turn("assistant", reply.text)]
        print(f"\n{reply.text}\n")
        if args.debug:
            used = ", ".join(s.doc_id for s in reply.sources) or "-"
            risk = reply.level.name.lower()
            print(f"  [{reply.mode} | risk={risk} | {used} | {reply.timings_ms}]\n")


def _cmd_eval(args: argparse.Namespace, cfg: Settings) -> int:
    from . import evaluate as ev
    from .bootstrap import build_retriever

    retriever = build_retriever(cfg, backend=args.backend)
    results = ev.run_all(
        retriever,
        ev.load_jsonl(Path(args.data) / "retrieval.jsonl"),
        ev.load_jsonl(Path(args.data) / "safety.jsonl"),
    )
    report = ev.render_markdown(results)
    if args.write:
        Path(args.write).write_text(report, encoding="utf-8")
        print(f"wrote {args.write}")
    else:
        print(report)
    return 0


def _cmd_kb_check(args: argparse.Namespace, cfg: Settings) -> int:
    from .kb import KBError, chunk_doc, load_docs

    problems: list[str] = []
    try:
        docs = load_docs(cfg.kb_dir)
    except KBError as exc:
        print(f"error: {exc}")
        return 1
    total = 0
    for doc in docs:
        chunks = chunk_doc(doc)
        total += len(chunks)
        words = len(doc.body.split())
        if not doc.topics:
            problems.append(f"{doc.id}: no topics")
        if words < 80:
            problems.append(f"{doc.id}: only {words} words")
        if not chunks:
            problems.append(f"{doc.id}: produced no chunks")
    print(f"{len(docs)} documents, {total} chunks")
    for p in problems:
        print(f"  - {p}")
    return 1 if problems else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mindbridge", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("serve", help="run the API and web UI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=_cmd_serve)

    p = sub.add_parser("chat", help="talk to the assistant in the terminal")
    p.add_argument("--debug", action="store_true", help="show mode, risk level and sources")
    p.set_defaults(func=_cmd_chat)

    p = sub.add_parser("eval", help="run retrieval and crisis-detection evaluation")
    p.add_argument("--backend", choices=["auto", "lsa", "sbert"], default=None)
    p.add_argument("--data", default="eval")
    p.add_argument("--write", metavar="PATH", help="write the markdown report here")
    p.set_defaults(func=_cmd_eval)

    p = sub.add_parser("kb-check", help="validate the knowledge base documents")
    p.set_defaults(func=_cmd_kb_check)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    return args.func(args, get_settings())


if __name__ == "__main__":
    sys.exit(main())
