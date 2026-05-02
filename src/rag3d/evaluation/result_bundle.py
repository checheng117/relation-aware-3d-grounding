"""Normalize multi-run metrics into a stable JSON + flat summary CSV schema."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def merge_main_stratified(
    runs: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Each run: {id, regime, main: {model_kind: {acc@1,...}}, stratified: {...}}."""
    main_out: dict[str, Any] = {}
    strat_out: dict[str, Any] = {}
    for r in runs:
        rid = str(r["id"])
        regime = str(r.get("regime", "unknown"))
        key = f"{regime}::{rid}"
        main_out[key] = r.get("main", {})
        strat_out[key] = r.get("stratified", {})
    return main_out, strat_out


def extract_hard_case_slice(stratified: dict[str, Any]) -> dict[str, Any]:
    """Pull subset:: keys into a compact dict per run key."""
    out: dict[str, Any] = {}
    for run_key, model_block in stratified.items():
        if not isinstance(model_block, dict):
            continue
        per_model: dict[str, Any] = {}
        for model_name, metrics in model_block.items():
            if not isinstance(metrics, dict):
                continue
            sub = {k: v for k, v in metrics.items() if "subset::" in k or "slice::" in k}
            if sub:
                per_model[model_name] = sub
        if per_model:
            out[run_key] = per_model
    return out


def summary_csv_rows(
    runs: list[dict[str, Any]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for r in runs:
        rid = str(r["id"])
        regime = str(r.get("regime", ""))
        main = r.get("main") or {}
        for model_name, m in main.items():
            if not isinstance(m, dict):
                continue
            rows.append(
                {
                    "run_id": rid,
                    "regime": regime,
                    "model": model_name,
                    "acc@1": str(m.get("acc@1", "")),
                    "acc@5": str(m.get("acc@5", "")),
                    "n": str(m.get("n", "")),
                }
            )
    return rows


def write_summary_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["run_id", "regime", "model", "acc@1", "acc@5", "n"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def build_main_comparison_table(
    runs: list[dict[str, Any]],
    row_ids: tuple[str, str, str, str] = ("A_baseline", "B_raw_relation", "C_structured", "D_hardneg"),
) -> list[dict[str, Any]]:
    """Expect runs with id in row_ids and regime entity|full; main single-key per model line."""

    def _find(regime: str, rid: str) -> dict[str, Any] | None:
        for r in runs:
            if r.get("regime") == regime and r.get("id") == rid:
                return r.get("main")
        return None

    lines = []
    labels = [
        ("A_baseline", "A attribute-only"),
        ("B_raw_relation", "B raw-text relation"),
        ("C_structured", "C structured relation-aware"),
        ("D_hardneg", "D structured + hard-negative"),
    ]
    for rid, label in labels:
        me = _find("entity", rid)
        mf = _find("full", rid)
        # Each main dict has one model key — take first values
        def acc(m: dict[str, Any] | None, k: str) -> str:
            if not m:
                return ""
            for _mk, mv in m.items():
                if isinstance(mv, dict) and k in mv:
                    return str(mv[k])
            return ""

        lines.append(
            {
                "row": label,
                "run_id": rid,
                "controlled_acc@1": acc(me, "acc@1"),
                "controlled_acc@5": acc(me, "acc@5"),
                "controlled_n": acc(me, "n"),
                "full_acc@1": acc(mf, "acc@1"),
                "full_acc@5": acc(mf, "acc@5"),
                "full_n": acc(mf, "n"),
            }
        )
    return lines


def write_main_table_csv(path: Path, lines: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "row",
        "run_id",
        "controlled_acc@1",
        "controlled_acc@5",
        "controlled_n",
        "full_acc@1",
        "full_acc@5",
        "full_n",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for line in lines:
            w.writerow({k: line.get(k, "") for k in fields})


def write_main_table_md(path: Path, lines: list[dict[str, Any]]) -> None:
    hdr = "| Model | Controlled Acc@1 | Controlled Acc@5 | Full Acc@1 | Full Acc@5 |\n|----|----:|----:|----:|----:|"
    body = []
    for line in lines:
        body.append(
            f"| {line['row']} | {line.get('controlled_acc@1', '')} | {line.get('controlled_acc@5', '')} | "
            f"{line.get('full_acc@1', '')} | {line.get('full_acc@5', '')} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(hdr + "\n" + "\n".join(body) + "\n", encoding="utf-8")


def hard_case_results_to_csv(hard_case: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run_key", "model", "metric", "value"])
        for run_key, per_model in hard_case.items():
            if not isinstance(per_model, dict):
                continue
            for model_name, metrics in per_model.items():
                if not isinstance(metrics, dict):
                    continue
                for mk, mv in metrics.items():
                    w.writerow([run_key, model_name, mk, mv])


def write_official_main_table_csv(path: Path, lines: list[dict[str, Any]]) -> None:
    fields = [
        "row",
        "run_id",
        "controlled_acc@1",
        "controlled_acc@5",
        "full_acc@1",
        "full_acc@5",
        "n_seeds",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for line in lines:
            w.writerow({k: str(line.get(k, "")) for k in fields})


def write_official_main_table_md(path: Path, lines: list[dict[str, Any]]) -> None:
    hdr = (
        "| Model | Controlled Acc@1 | Controlled Acc@5 | Full Acc@1 | Full Acc@5 | n_seeds | notes |\n"
        "|----|----:|----:|----:|----:|---:|---|"
    )
    body = []
    for line in lines:
        body.append(
            f"| {line['row']} | {line.get('controlled_acc@1', '')} | {line.get('controlled_acc@5', '')} | "
            f"{line.get('full_acc@1', '')} | {line.get('full_acc@5', '')} | {line.get('n_seeds', '')} | "
            f"{line.get('notes', '')} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(hdr + "\n" + "\n".join(body) + "\n", encoding="utf-8")


def _mean_std(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return float("nan"), float("nan")
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
    return m, var**0.5


def build_official_main_lines(
    seed_metrics: dict[int, list[dict[str, Any]]],
    row_specs: list[tuple[str, str]],
    seeds: list[int],
) -> list[dict[str, Any]]:
    """seed_metrics[seed] = list of {id, regime, main} per eval job."""

    def _acc_from_main(main: dict[str, Any], k: str) -> float | None:
        if not main:
            return None
        for _mk, mv in main.items():
            if isinstance(mv, dict) and k in mv:
                try:
                    return float(mv[k])
                except (TypeError, ValueError):
                    return None
        return None

    lines: list[dict[str, Any]] = []
    n_seeds = len(seeds)
    for rid, label in row_specs:
        ent_vals1: list[float] = []
        ent_vals5: list[float] = []
        full_vals1: list[float] = []
        full_vals5: list[float] = []
        for s in seeds:
            runs = seed_metrics.get(s) or []
            for r in runs:
                if r.get("id") != rid:
                    continue
                main = r.get("main") or {}
                if r.get("regime") == "entity":
                    v1 = _acc_from_main(main, "acc@1")
                    v5 = _acc_from_main(main, "acc@5")
                    if v1 is not None:
                        ent_vals1.append(v1)
                    if v5 is not None:
                        ent_vals5.append(v5)
                elif r.get("regime") == "full":
                    v1 = _acc_from_main(main, "acc@1")
                    v5 = _acc_from_main(main, "acc@5")
                    if v1 is not None:
                        full_vals1.append(v1)
                    if v5 is not None:
                        full_vals5.append(v5)

        def fmt(vs: list[float]) -> str:
            if not vs:
                return ""
            if len(vs) == 1:
                return f"{vs[0]:.6f}"
            m, sd = _mean_std(vs)
            return f"{m:.6f} ± {sd:.6f}"

        note = f"seeds {seeds}" if n_seeds else ""
        lines.append(
            {
                "row": label,
                "run_id": rid,
                "controlled_acc@1": fmt(ent_vals1),
                "controlled_acc@5": fmt(ent_vals5),
                "full_acc@1": fmt(full_vals1),
                "full_acc@5": fmt(full_vals5),
                "n_seeds": str(n_seeds),
                "notes": note,
            }
        )
    return lines
