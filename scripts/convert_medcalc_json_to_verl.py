#!/usr/bin/env python3
"""Convert MedCalc instruction/input/output JSON data to verl parquet format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_records(path: Path) -> list[dict[str, Any]]:
    """Load either a JSON array file or a JSONL file."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if text[0] == "[":
        records = json.loads(text)
        if not isinstance(records, list):
            raise ValueError(f"Expected a JSON list in {path}")
        return records

    records = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc
        records.append(item)
    return records


def build_prompt(instruction: str, input_text: str) -> str:
    instruction = instruction.strip()
    input_text = input_text.strip()
    if instruction and input_text:
        return f"{instruction}\n\n{input_text}"
    return instruction or input_text


def convert_records(records: list[dict[str, Any]], data_source: str, ability: str) -> pd.DataFrame:
    rows = []
    for idx, item in enumerate(records):
        if not isinstance(item, dict):
            raise ValueError(f"Record {idx} is not an object: {type(item).__name__}")

        instruction = str(item.get("instruction", ""))
        input_text = str(item.get("input", ""))
        output = str(item.get("output", ""))
        prompt_text = build_prompt(instruction, input_text)

        if not prompt_text:
            raise ValueError(f"Record {idx} has empty instruction/input fields.")

        rows.append(
            {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": prompt_text}],
                "ability": ability,
                "reward_model": {
                    "ground_truth": output,
                    "style": "reference_output",
                },
                "extra_info": {
                    "index": idx,
                    "instruction": instruction,
                    "input": input_text,
                    "reference_output": output,
                },
            }
        )

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert MedCalc instruction/input/output JSON data into verl parquet format."
    )
    parser.add_argument(
        "--input",
        default="/czsun/zhi/xywang/anchored_learning/LlamaFactory/data/medcalc_train.json",
        help="Path to the source JSON or JSONL file.",
    )
    parser.add_argument(
        "--output",
        default="datasets/medcalc_train.parquet",
        help="Path to write the verl-format parquet file.",
    )
    parser.add_argument("--data-source", default="medcalc", help="Value for the data_source column.")
    parser.add_argument("--ability", default="medical_calculation", help="Value for the ability column.")
    parser.add_argument("--preview", type=int, default=2, help="Number of converted samples to print.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    records = load_records(input_path)
    if not records:
        raise ValueError(f"No records found in {input_path}")

    df = convert_records(records, data_source=args.data_source, ability=args.ability)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    print(f"Converted {len(df)} records")
    print(f"Wrote: {output_path}")
    print(f"Columns: {list(df.columns)}")
    for idx, row in df.head(args.preview).iterrows():
        prompt = row["prompt"][0]["content"]
        print(f"\n[preview {idx}] prompt: {prompt[:500]}")
        print(f"[preview {idx}] ground_truth: {row['reward_model']['ground_truth'][:300]}")


if __name__ == "__main__":
    main()
