"""Thin CLI entry (train/eval wiring lives in `scripts/` for research workflows)."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="rag3d", description="Relation-aware 3D grounding package")
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("version", help="print version")
    v.set_defaults(func=_cmd_version)
    args = p.parse_args(argv)
    args.func(args)


def _cmd_version(_: argparse.Namespace) -> None:
    from rag3d import __version__

    print(__version__)


if __name__ == "__main__":
    main()
