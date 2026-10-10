# Report LLM labeling

This folder labels **radiology report text** for the GI-bleed cohort. It does not read images. Copy the folder; it does not need the rest of the training repository.

The output columns match the earlier Abdominal_Trauma labeling CSVs (`llm_labeled_reports_*.csv`). Those files are not in this repository.

```bash
pip install -r requirements.txt
```

Needs an OpenAI-compatible chat server (vLLM, Ollama, Together, or OpenAI). Python 3.10+.

## What it labels

| Task | `--task` | Output columns | Use |
|---|---|---|---|
| Injury / extravasation | `injury` | `Bowel`, `Kidney`, `Liver`, `Extravasation`, `Explanation` | Split the 6k CTA pull into bleed vs explicit-negative |
| Bleed site | `location` | `BOWELS`, `STOMACH`, `ESOPHAGUS`, `LIVER`, … | Site flags on reports already called positive |

Allowed values are `Yes`, `No`, `Maybe`. A parse or server failure is written as `Failed` and retried on the next run. An empty report is `Undefined` and is not sent again.

**Positive exam:** `Extravasation=Yes` (active blush / active GI bleed).  
**Control pool:** `Extravasation=No` and the report text actually denies active bleed. Do not treat a missing GSPS box as a control. Extra-GI bleeding stays out of the control pool.

## Input

CSV or Excel with at least:

- report text (`text`, `Report Text`, `impression`, …)
- accession (`Accession Number`, `Accession`, `acc`)
- optional MRN (`Patient MRN`, `MRN`)

The MGB pull `GI_Bleed_Extravasation.xlsx` works as `--input`.

## How to run it

Local vLLM example:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=dummy

python label_reports.py \
  --input /data/site/reports.xlsx \
  --task injury \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --out /data/site/llm_injury.csv
```

Then sites. If the input has an `Extravasation` column, only `Yes` rows are sent:

```bash
python label_reports.py \
  --input /data/site/llm_injury.csv \
  --task location \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --out /data/site/llm_location.csv
```

`--out` is append/resume-safe. Re-run the same command after a drop: finished accessions are skipped, and `Failed` rows are removed so they are tried again. Excel accessions like `12345.0` are stored as `12345`.

Merge two models and dump disagreements for a person to read:

```bash
python merge_labels.py \
  --inputs llama.csv qwen.csv \
  --out merged.csv \
  --review review.csv
```

Injury columns are `Bowel`, `Kidney`, `Liver`, `Extravasation`, plus `Explanation`, `text`, `Accession Number`, and `Patient MRN`. Location columns are the ten site flags in `prompts.py`, plus the same id columns. These labels are report text, not the CT model scores from `inference/`.
