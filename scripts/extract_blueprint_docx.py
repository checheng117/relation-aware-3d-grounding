#!/usr/bin/env python3
"""Extract plain text from the project blueprint .docx (WordprocessingML)."""

from __future__ import annotations

import argparse
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def paragraph_text(p: ET.Element) -> str:
    parts: list[str] = []
    for t in p.findall(".//w:t", NS):
        if t.text:
            parts.append(t.text)
        if t.tail:
            parts.append(t.tail)
    return "".join(parts)


def extract_docx(path: Path) -> str:
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml")
    root = ET.fromstring(xml)
    lines: list[str] = []
    for p in root.findall(".//w:p", NS):
        t = paragraph_text(p).strip()
        if t:
            lines.append(t)
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "docx",
        type=Path,
        nargs="?",
        default=Path("docs/CSC6133_Upgraded_3D_Spatial_Reasoning_Blueprint_CheCheng.docx"),
    )
    ap.add_argument("-o", "--out", type=Path, default=None)
    args = ap.parse_args()
    text = extract_docx(args.docx)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text + "\n")


if __name__ == "__main__":
    main()
