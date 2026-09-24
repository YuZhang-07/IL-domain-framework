#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare the Task 2 remove_hbonds CSV without relabeling or rescaling.

Default deletion rule: configs/task2/evaluation/ablation/remove_hbonds.json.
The input file is never changed. Output must be a new filename.
"""
from __future__ import annotations

import json
import re
import pandas as pd
import sys
from pathlib import Path
_EVAL_DIR = Path(__file__).resolve().parent.parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from _eval_runtime import ablation_args, public_path, save_preparation_info


def main(argv=None):
    args = ablation_args("remove_hbonds", "prepare", argv)
    if args.dry_run:
        print(json.dumps(args.effective_config, ensure_ascii=False, indent=2))
        return
    spec = args.config_values["preparation"]
    df = pd.read_csv(args.data_path)
    if spec["mode"] == "explicit":
        columns_to_remove = list(spec["columns"])
        missing = [col for col in columns_to_remove if col not in df.columns]
        if missing:
            raise ValueError(f"Required removal columns are missing: {missing}")
    else:
        pattern = re.compile(spec["pattern"])
        columns_to_remove = [col for col in df.columns if pattern.match(col)]
        if not columns_to_remove:
            raise ValueError("No columns match the supplied residue-pair pattern.")
        if len(columns_to_remove) != spec["expected_count"]:
            print(f"Warning: found {len(columns_to_remove)} matching columns; expected {spec['expected_count']}. Original warning-only behavior is retained.")
    result = df.drop(columns=columns_to_remove)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create guards against concurrent runs overwriting the same file.
    with args.output_path.open("x", encoding="utf-8", newline="") as handle:
        result.to_csv(handle, index=False)
    save_preparation_info(args, __file__, columns_to_remove, len(result), result.shape[1])
    print(f"Input: {public_path(args.data_path)}")
    print(f"Output: {public_path(args.output_path)}")
    print(f"Removed {len(columns_to_remove)} columns; remaining columns={result.shape[1]}, rows={len(result)}.")


if __name__ == "__main__":
    main()
