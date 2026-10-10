#!/usr/bin/env python3
"""Label radiology reports for active GI bleed / extravasation.

Talks to any OpenAI-compatible chat endpoint (vLLM, Ollama, Together, OpenAI).

    export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
    export OPENAI_API_KEY=dummy
    python label_reports.py --input reports.csv --task injury --model llama70b --out out/injury.csv

Input needs a report-text column and an accession column. An Excel pull
(``GI_Bleed_Extravasation.xlsx``) works if those headers exist.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI

from prompts import (
    ALLOWED,
    INJURY_KEYS,
    INJURY_SYSTEM,
    INJURY_USER,
    LOCATION_KEYS,
    LOCATION_SYSTEM,
    LOCATION_USER,
)

TEXT_CANDIDATES = (
    "text",
    "Report Text",
    "report",
    "Impression",
    "impression",
    "Report",
    "NARRATIVE",
)
ACC_CANDIDATES = ("Accession Number", "Accession", "accession", "acc")
MRN_CANDIDATES = ("Patient MRN", "MRN", "mrn", "PatientID")


def _cell(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    s = str(v).strip()
    if s.lower() in {"nan", "none"}:
        return ""
    if re.fullmatch(r"\d+\.0+", s):
        s = s.split(".", 1)[0]
    return s


def _pick(df: pd.DataFrame, names: tuple[str, ...]) -> str:
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n in df.columns:
            return n
        if n.lower() in lower:
            return lower[n.lower()]
    raise SystemExit(f"Need one of {names}; got {list(df.columns)}")


def _load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def _parse_json(raw: str) -> dict:
    s = (raw or "").strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.I)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", s, flags=re.S)
        if m:
            return json.loads(m.group(0))
        raise


def _norm_flag(v) -> str:
    s = str(v or "").strip()
    if s in ALLOWED:
        return s
    low = s.lower()
    if low in {"yes", "y", "true", "1"}:
        return "Yes"
    if low in {"no", "n", "false", "0"}:
        return "No"
    if low in {"maybe", "uncertain", "possible", "unclear"}:
        return "Maybe"
    return "Undefined"


def _task_keys(task: str) -> tuple[str, ...]:
    return LOCATION_KEYS if task == "location" else INJURY_KEYS


def label_one(client: OpenAI, model: str, task: str, report: str, max_tokens: int) -> dict:
    if task == "location":
        sys_p, user_p, keys = LOCATION_SYSTEM, LOCATION_USER, LOCATION_KEYS
    else:
        sys_p, user_p, keys = INJURY_SYSTEM, INJURY_USER, INJURY_KEYS
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": sys_p},
                    {"role": "user", "content": user_p.format(report=report)},
                ],
            )
            raw = (resp.choices[0].message.content or "").strip()
            blob = _parse_json(raw)
            out = {k: _norm_flag(blob.get(k)) for k in keys}
            if task == "injury":
                out["Explanation"] = str(blob.get("Explanation") or "").strip()
            out["raw_text"] = raw
            return out
        except Exception as e:
            last_err = e
            if attempt < 2:
                time.sleep(2.0 * (attempt + 1))
    assert last_err is not None
    raise last_err


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="CSV or XLSX of reports")
    p.add_argument("--out", required=True, help="Output CSV (resume-safe)")
    p.add_argument("--task", choices=("injury", "location"), default="injury")
    p.add_argument("--model", default=os.environ.get("LLM_MODEL", "llama70b"))
    p.add_argument("--text-col", default="")
    p.add_argument("--acc-col", default="")
    p.add_argument("--mrn-col", default="")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--sleep", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    args = p.parse_args()

    src = _load_table(Path(args.input))
    text_c = args.text_col or _pick(src, TEXT_CANDIDATES)
    acc_c = args.acc_col or _pick(src, ACC_CANDIDATES)
    mrn_c = args.mrn_col or None
    if not mrn_c:
        try:
            mrn_c = _pick(src, MRN_CANDIDATES)
        except SystemExit:
            mrn_c = None

    if args.task == "location" and "Extravasation" in src.columns:
        before = len(src)
        src = src[src["Extravasation"].map(_norm_flag) == "Yes"].copy()
        print(f"location task: {len(src)} of {before} rows with Extravasation=Yes", flush=True)

    keys = _task_keys(args.task)
    done: set[str] = set()
    out_path = Path(args.out)
    if out_path.is_file() and out_path.stat().st_size:
        prev = pd.read_csv(out_path, dtype=str, keep_default_na=False)
        if "Accession Number" not in prev.columns:
            raise SystemExit(f"{out_path} has no Accession Number column")
        prev["Accession Number"] = prev["Accession Number"].map(_cell)
        present = [k for k in keys if k in prev.columns]
        failed = prev[present].eq("Failed").any(axis=1) if present else pd.Series(False, index=prev.index)
        if failed.any():
            prev = prev.loc[~failed].copy()
            prev.to_csv(out_path, index=False)
            print(f"dropped {int(failed.sum())} failed rows so this run can retry them", flush=True)
        done = set(prev["Accession Number"])
        print(f"resume {len(done)} already in {out_path}", flush=True)

    rows = src.to_dict("records")
    if args.limit:
        rows = rows[: int(args.limit)]

    if args.task == "location":
        fields = list(LOCATION_KEYS) + ["text", "Accession Number", "Patient MRN", "raw_text"]
    else:
        fields = list(INJURY_KEYS) + ["Explanation", "text", "Accession Number", "Patient MRN", "raw_text"]

    client = OpenAI(base_url=args.base_url, api_key=args.api_key or "empty")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.is_file() or out_path.stat().st_size == 0
    n_ok = n_fail = 0
    with out_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for i, rec in enumerate(rows, 1):
            acc = _cell(rec.get(acc_c, ""))
            if not acc or acc in done:
                continue
            done.add(acc)
            text = _cell(rec.get(text_c, ""))
            if not text:
                n_fail += 1
                lab = {k: "Undefined" for k in keys}
                lab["Explanation"] = ""
                lab["raw_text"] = "ERROR: empty report"
            else:
                try:
                    lab = label_one(client, args.model, args.task, text, args.max_tokens)
                    n_ok += 1
                except Exception as e:
                    n_fail += 1
                    lab = {k: "Failed" for k in keys}
                    lab["Explanation"] = ""
                    lab["raw_text"] = f"ERROR: {type(e).__name__}: {e}"
            lab["text"] = text
            lab["Accession Number"] = acc
            lab["Patient MRN"] = _cell(rec.get(mrn_c, "")) if mrn_c else ""
            w.writerow({k: lab.get(k, "") for k in fields})
            f.flush()
            if i % 25 == 0:
                print(f"[{i}/{len(rows)}] ok={n_ok} fail={n_fail}", flush=True)
            if args.sleep:
                time.sleep(args.sleep)
    print(f"wrote {out_path} ok={n_ok} fail={n_fail}", flush=True)


if __name__ == "__main__":
    main()
