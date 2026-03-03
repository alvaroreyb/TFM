"""
This script runs one or more Hugging Face token-classification NER models over a CSV file and exports all extracted
entities into a single CSV.

Privacy constraints implemented:
- The output does not include any original CSV column names.
- The output does not include the original text nor any original identifier columns.
- Only row indices (row_id) are used to reference documents.

Multi-model behavior:
- Pass one or more model IDs via --model_ids (comma-separated). All models are executed and their entities aggregated.
"""

import os
import argparse
from datetime import datetime
from typing import List, Dict, Any

import pandas as pd
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


def pick_device_id() -> int:
    """
    Selects a device id for Hugging Face pipeline.
    Arguments: None.
    Return: 0 if CUDA is usable, otherwise -1 for CPU.
    """
    if torch.cuda.is_available():
        try:
            _ = torch.tensor([0.0]).cuda()
            return 0
        except Exception:
            pass
    return -1


def parse_model_ids(model_ids: str) -> List[str]:
    """
    Parses a comma-separated list of model IDs.
    Arguments:
        model_ids: Comma-separated string of model ids.
    Return:
        List of model ids.
    """
    parts = [p.strip() for p in model_ids.split(",")]
    return [p for p in parts if p]


def resolve_text_column(df: pd.DataFrame, text_col_arg: str) -> str:
    """
    Resolves the text column to use for inference.
    Arguments:
        df: Input dataframe.
        text_col_arg: User-provided column name (possibly empty).
    Return:
        The chosen column name.
    """
    if text_col_arg:
        if text_col_arg not in df.columns:
            raise ValueError("Provided text column not found in the CSV.")
        return text_col_arg

    candidates = ["TEXTO_LIMPIO", "TEXTO", "TEXT", "text", "note", "NOTA", "NOTES"]
    for c in candidates:
        if c in df.columns:
            return c

    raise ValueError("Text column not found. Provide it via --text_col.")


def build_ner_pipeline(model_id: str, device_id: int, batch_size: int):
    """
    Builds a Hugging Face token-classification pipeline for a given model id.
    Arguments:
        model_id: Hugging Face model id.
        device_id: -1 for CPU, >=0 for CUDA device index.
        batch_size: Pipeline batch size.
    Return:
        A configured transformers pipeline.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForTokenClassification.from_pretrained(model_id)
    return pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=device_id,
        batch_size=batch_size
    )


def normalize_entity(item: Dict[str, Any], model_id: str, row_id: str) -> Dict[str, Any]:
    """
    Normalizes a single NER output item into a flat dict row for export.
    Arguments:
        item: Raw entity dict from transformers pipeline.
        model_id: Model identifier used for inference.
        row_id: String row identifier (derived from dataframe index).
    Return:
        A normalized dict with stable keys for export.
    """
    return {
        "run_id": None,
        "model_id": model_id,
        "row_id": row_id,
        "label": item.get("entity_group") or item.get("entity"),
        "span": str(item.get("word", "")).strip(),
        "start": int(item.get("start", -1)),
        "end": int(item.get("end", -1)),
        "score": float(item.get("score", 0.0)),
    }


def main():
    parser = argparse.ArgumentParser(description="Multimodel NER extraction from CSV with privacy-preserving outputs.")
    parser.add_argument(
        "--input_csv",
        type=str,
        default=r""
    )
    parser.add_argument("--text_col", type=str, default="")
    parser.add_argument(
        "--model_ids",
        type=str,
        default="OpenMed/OpenMed-NER-OncologyDetect-SuperMedical-355M",
        help="Comma-separated Hugging Face model IDs."
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--score_min_entity_save",
        type=float,
        default=0.85,
        help="Minimum score to store an entity in the output CSV. Use 0 to store all."
    )
    parser.add_argument("--max_docs", type=int, default=0, help="If >0, limit to N documents.")
    parser.add_argument(
        "--output_csv",
        type=str,
        default=r""
    )
    args = parser.parse_args()

    out_dir = os.path.dirname(args.output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(args.input_csv, dtype=str)
    text_col = resolve_text_column(df, args.text_col)

    texts = df[text_col].fillna("").tolist()
    if args.max_docs and args.max_docs > 0:
        texts = texts[:args.max_docs]
        df = df.iloc[:args.max_docs].copy()

    row_ids = df.index.astype(str).tolist()

    device_id = pick_device_id()
    device_name = "GPU" if device_id >= 0 else "CPU"
    model_id_list = parse_model_ids(args.model_ids)

    run_id = datetime.today().strftime("%Y-%m-%d_%H%M%S")

    rows: List[Dict[str, Any]] = []

    for model_id in model_id_list:
        print(f"Loading NER model: {model_id} (device={device_name})")
        ner = build_ner_pipeline(model_id, device_id=device_id, batch_size=args.batch_size)

        print(f"Running NER on {len(texts)} documents for model: {model_id}")
        for i in tqdm(range(len(texts))):
            ents = ner(texts[i])

            if args.score_min_entity_save > 0:
                ents = [e for e in ents if float(e.get("score", 0.0)) >= args.score_min_entity_save]

            for e in ents:
                rec = normalize_entity(e, model_id=model_id, row_id=row_ids[i])
                rec["run_id"] = run_id
                rows.append(rec)

    df_entities = pd.DataFrame(rows, columns=["run_id", "model_id", "row_id", "label", "span", "start", "end", "score"])
    df_entities.to_csv(args.output_csv, index=False, encoding="utf-8")
    print(f"[Saved] Entities -> {args.output_csv}")
    print("Done.")


if __name__ == "__main__":
    main()
