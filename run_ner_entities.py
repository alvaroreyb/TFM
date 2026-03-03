"""
This script runs a Spanish oncology NER model over a CSV dataset, extracts entities, and produces a
document-level recurrence prediction using simple keyword + negation heuristics. It saves:
- entities.csv with extracted entities
- label_counts.json with entity label frequencies
- label_counts_by_class.csv if a ground-truth RECURRENCIA column exists
- doc_predictions.csv with per-document predictions
- metrics.json and optional plots when ground-truth labels exist
"""

import os
import re
import json
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional

import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


RECURRENCE_KEYWORDS = [
    "recurrencia", "recaida", "recaída", "metastasis", "metástasis",
    "progresion", "progresión", "reaparicion", "reaparición", "reincidencia"
]

NEGATION_PATTERNS = [
    r"\bno\s+(?:hay\s+)?(?:datos|evidencia|signos|sospecha)\s+(?:de\s+)?recurr",
    r"\bsin\s+(?:evidencia|datos|signos|hallazgos)?\s*(?:de\s+)?recurr",
    r"\bdescarta\s+recurr",
    r"\bse\s+descarta\s+recurr",
    r"\bnegativ[oa](?:\s+para)?\s+recurr",
    r"\btriple\s+negativ",
    r"\bno\s+recurr",
    r"\bno\s+recaid",
    r"\bsin\s+recurr",
    r"\b(?:ausencia|libre)\s+de\s+recurr",
    r"\bno\s+se\s+objetiva\s+recurr",
    r"\bno\s+hallazgos\s+de\s+recurr",
    r"\btratad[ao]\s+sin\s+recurr",
    r"\bnormal\s+sin\s+datos\s+de\s+recurr",
    r"\bno\s+acude\b",
]

NEGATION_REGEX = re.compile("|".join(NEGATION_PATTERNS), flags=re.IGNORECASE)


def safe_name(s: str) -> str:
    """
    Makes a Windows-friendly folder/file name from an arbitrary string.
    Arguments: s (str).
    Return: str.
    """
    s = s.replace("/", "-").replace("\\", "-")
    s = re.sub(r"[^A-Za-z0-9._\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def pick_device_id() -> int:
    """
    Selects a valid Transformers pipeline device id.
    Arguments: none.
    Return: int (-1 for CPU, 0 for first CUDA GPU).
    """
    if torch.cuda.is_available():
        try:
            _ = torch.tensor([0.0]).cuda()
            return 0
        except Exception:
            return -1
    return -1


def clean_ner_item(
    item: Dict[str, Any],
    text_id: Optional[str] = None,
    fold: Optional[str] = None,
    patient_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Normalizes a Transformers NER pipeline item into a flat record for CSV export.
    Arguments: item, text_id, fold, patient_id.
    Return: Dict[str, Any].
    """
    return {
        "text_id": text_id,
        "fold": fold,
        "patient_id": patient_id,
        "label": item.get("entity_group") or item.get("entity"),
        "span": str(item.get("word", "")).strip(),
        "start": int(item.get("start", -1)),
        "end": int(item.get("end", -1)),
        "score": float(item.get("score", 0.0)),
    }


def text_has_recurrence(text: str) -> bool:
    """
    Checks if the text contains any recurrence-related keyword.
    Arguments: text (str).
    Return: bool.
    """
    if not isinstance(text, str) or not text:
        return False
    tl = text.lower()
    return any(kw in tl for kw in RECURRENCE_KEYWORDS)


def text_is_negated(text: str) -> bool:
    """
    Checks whether recurrence appears negated in the text using regex patterns.
    Arguments: text (str).
    Return: bool.
    """
    if not isinstance(text, str) or not text:
        return False
    return bool(NEGATION_REGEX.search(text.lower()))


def filter_relevant_entities(
    entities: List[Dict[str, Any]],
    score_threshold: float
) -> List[Dict[str, Any]]:
    """
    Filters entities to those containing recurrence keywords and meeting a score threshold.
    Arguments: entities (List[Dict[str, Any]]), score_threshold (float).
    Return: List[Dict[str, Any]].
    """
    out = []
    for ent in entities:
        word = str(ent.get("word", "")).lower()
        score = float(ent.get("score", 0.0))
        if score >= score_threshold and any(kw in word for kw in RECURRENCE_KEYWORDS):
            out.append(ent)
    return out


def decide_doc_positive(
    text: str,
    entities: List[Dict[str, Any]],
    keyword_score_threshold: float
) -> bool:
    """
    Produces a document-level recurrence prediction based on relevant entities and negation.
    Arguments: text (str), entities (List[Dict[str, Any]]), keyword_score_threshold (float).
    Return: bool.
    """
    relevant = filter_relevant_entities(entities, keyword_score_threshold)
    if not relevant:
        return False
    if text_is_negated(text):
        return False
    return True


def resolve_text_column(df: pd.DataFrame, text_col_arg: str) -> str:
    """
    Resolves which column contains document text, using either a provided name or defaults.
    Arguments: df (pd.DataFrame), text_col_arg (str).
    Return: str.
    """
    if text_col_arg:
        if text_col_arg not in df.columns:
            raise ValueError(f"Column '{text_col_arg}' not found in CSV.")
        return text_col_arg

    if "TEXTO_LIMPIO" in df.columns:
        return "TEXTO_LIMPIO"
    if "TEXTO" in df.columns:
        return "TEXTO"
    raise ValueError("Neither 'TEXTO_LIMPIO' nor 'TEXTO' found. Use --text_col.")


def build_output_dir(output_root: str, model_id: str) -> str:
    """
    Creates and returns an output directory path based on date and model id.
    Arguments: output_root (str), model_id (str).
    Return: str.
    """
    today = datetime.today().strftime("%Y-%m-%d")
    safe_model = safe_name(model_id)
    out_dir = os.path.join(output_root, f"{today}_{safe_model}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def run_ner_over_texts(
    ner_pipe,
    texts: List[str],
    score_min: float
) -> List[List[Dict[str, Any]]]:
    """
    Runs NER over a list of texts and applies a minimum score filter.
    Arguments: ner_pipe, texts (List[str]), score_min (float).
    Return: List[List[Dict[str, Any]]].
    """
    all_entities = []
    for i in tqdm(range(len(texts))):
        ents = ner_pipe(texts[i])
        if score_min > 0:
            ents = [e for e in ents if float(e.get("score", 0.0)) >= score_min]
        all_entities.append(ents)
    return all_entities


def save_label_counts(out_dir: str, df_entities: pd.DataFrame) -> Dict[str, int]:
    """
    Computes and saves entity label counts.
    Arguments: out_dir (str), df_entities (pd.DataFrame).
    Return: Dict[str, int].
    """
    if df_entities.empty:
        label_counts = {}
    else:
        label_counts = df_entities["label"].value_counts().to_dict()

    path = os.path.join(out_dir, "label_counts.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(label_counts, f, indent=2, ensure_ascii=False)

    return label_counts


def maybe_save_label_counts_by_class(out_dir: str, df: pd.DataFrame, df_entities: pd.DataFrame) -> None:
    """
    If RECURRENCIA exists, saves entity label counts stratified by class.
    Arguments: out_dir (str), df (pd.DataFrame), df_entities (pd.DataFrame).
    Return: None.
    """
    if "RECURRENCIA" not in df.columns or df_entities.empty:
        return

    doc_label = df[["RECURRENCIA"]].copy()
    doc_label["text_id"] = df.index.astype(str)
    merged = df_entities.merge(doc_label, on="text_id", how="left")
    by_class = merged.groupby(["RECURRENCIA", "label"]).size().reset_index(name="count")
    path = os.path.join(out_dir, "label_counts_by_class.csv")
    by_class.to_csv(path, index=False, encoding="utf-8")
    print(f"[Saved] Label counts by class -> {path}")


def compute_and_save_metrics(out_dir: str, y_true: List[int], y_pred: List[int]) -> List[List[int]]:
    """
    Computes and saves classification metrics and returns the confusion matrix.
    Arguments: out_dir (str), y_true (List[int]), y_pred (List[int]).
    Return: List[List[int]].
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "confusion_matrix": cm.tolist(),
    }

    path = os.path.join(out_dir, "metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("[Metrics]", metrics)
    return cm.tolist()


def maybe_save_plots(out_dir: str, cm: List[List[int]]) -> None:
    """
    Saves a confusion matrix heatmap plot.
    Arguments: out_dir (str), cm (List[List[int]]).
    Return: None.
    """
    plt.figure(figsize=(4, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Recurr.", "Recurr."],
        yticklabels=["No Recurr.", "Recurr."],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (NER-based decision)")
    plt.tight_layout()
    path = os.path.join(out_dir, "cm_ner_based.png")
    plt.savefig(path)
    plt.close()
    print(f"[Saved] Confusion matrix plot -> {path}")


def main() -> None:
    """
    Entry point: loads data, runs NER, saves outputs, and optionally evaluates vs RECURRENCIA.
    Arguments: none.
    Return: None.
    """
    parser = argparse.ArgumentParser(description="NER-based recurrence detection (Spanish breast cancer).")
    parser.add_argument(
        "--input_csv",
        type=str,
        default=r"C:\Users\Usuario\Desktop\PenVEC\TFM\TFM\data\recurrence_data_filtrado.csv",
    )
    parser.add_argument("--text_col", type=str, default="")
    parser.add_argument("--id_col", type=str, default="ID_PACIENTE")
    parser.add_argument("--fold_col", type=str, default="FOLD")
    parser.add_argument("--model_id", type=str, default="OpenMed/OpenMed-NER-OncologyDetect-SuperMedical-355M")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--score_min", type=float, default=0.95)
    parser.add_argument("--keyword_score_threshold", type=float, default=0.34)
    parser.add_argument("--max_docs", type=int, default=0)
    parser.add_argument("--output_root", type=str, default="../ner_outputs")
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv, dtype=str)
    text_col = resolve_text_column(df, args.text_col)

    has_label = "RECURRENCIA" in df.columns
    if has_label:
        df["RECURRENCIA"] = df["RECURRENCIA"].astype(str)

    out_dir = build_output_dir(args.output_root, args.model_id)

    device_id = pick_device_id()
    print(f"Loading NER model: {args.model_id} (device={'GPU' if device_id >= 0 else 'CPU'})")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForTokenClassification.from_pretrained(args.model_id)
    ner_pipe = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=device_id,
        batch_size=args.batch_size,
    )

    texts = df[text_col].fillna("").tolist()
    if args.max_docs and args.max_docs > 0:
        df = df.iloc[: args.max_docs].copy()
        texts = texts[: args.max_docs]

    text_ids = df.index.astype(str).tolist()
    patient_ids = df[args.id_col].tolist() if args.id_col in df.columns else [None] * len(df)
    folds = df[args.fold_col].tolist() if args.fold_col in df.columns else [None] * len(df)

    print(f"Running NER on {len(texts)} documents...")
    all_entities = run_ner_over_texts(ner_pipe, texts, args.score_min)

    rows = []
    doc_preds = []
    for i in range(len(texts)):
        ents = all_entities[i]
        for e in ents:
            rows.append(clean_ner_item(e, text_id=text_ids[i], fold=folds[i], patient_id=patient_ids[i]))
        pred_pos = decide_doc_positive(texts[i], ents, args.keyword_score_threshold)
        doc_preds.append(int(pred_pos))

    df_entities = pd.DataFrame(rows)
    entities_csv = os.path.join(out_dir, "entities.csv")
    df_entities.to_csv(entities_csv, index=False, encoding="utf-8")
    print(f"[Saved] Entities -> {entities_csv}")

    save_label_counts(out_dir, df_entities)
    maybe_save_label_counts_by_class(out_dir, df, df_entities)

    df_doc = pd.DataFrame(
        {
            "text_id": text_ids,
            "patient_id": patient_ids,
            "fold": folds,
            "pred_recurrence_ner": doc_preds,
        }
    )
    if has_label:
        df_doc["RECURRENCIA"] = df["RECURRENCIA"].astype(int).tolist()

    doc_pred_csv = os.path.join(out_dir, "doc_predictions.csv")
    df_doc.to_csv(doc_pred_csv, index=False, encoding="utf-8")
    print(f"[Saved] Document-level predictions -> {doc_pred_csv}")

    if has_label:
        y_true = df["RECURRENCIA"].astype(int).tolist()
        y_pred = doc_preds
        cm = compute_and_save_metrics(out_dir, y_true, y_pred)
        if args.plots:
            maybe_save_plots(out_dir, cm)

    print("Done.")


if __name__ == "__main__":
    main()
