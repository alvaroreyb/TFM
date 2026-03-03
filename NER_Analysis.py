"""
NER entity analysis script.
Generates summary statistics CSVs and interactive HTML charts from a NER predictions CSV.
Input columns: text_id, fold, patient_id, label, span, start, end, score
Outputs: summary_stats.csv, label_stats.csv, fold_stats.csv,
         top10_mentions_per_label.csv, label_frequency_table.csv, 8x HTML charts
"""

import os
import unicodedata
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "plotly_white"


CSV_PATH = 
OUTPUT_DIR =

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
df["span_length"] = df["end"] - df["start"]

print(f"Rows: {len(df)}, Patients: {df['patient_id'].nunique()}, Texts: {df['text_id'].nunique()}")


def normalize_span(text):
    """
    Normalizes a span string for grouping purposes.
    Arguments: text (str) - raw span string.
    Return: str - lowercased, accent-stripped, whitespace-trimmed string.
    """
    text = str(text).lower().strip()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text


df["span_normalized"] = df["span"].apply(normalize_span)

summary = {
    "total_entities": len(df),
    "unique_patients": df["patient_id"].nunique(),
    "unique_texts": df["text_id"].nunique(),
    "unique_labels": df["label"].nunique(),
    "mean_entities_per_text": round(df.groupby("text_id").size().mean(), 2),
    "std_entities_per_text": round(df.groupby("text_id").size().std(), 2),
    "mean_score": round(df["score"].mean(), 4),
    "std_score": round(df["score"].std(), 4),
    "pct_low_confidence_05": round((df["score"] < 0.5).mean() * 100, 2),
    "pct_low_confidence_07": round((df["score"] < 0.7).mean() * 100, 2),
}
pd.DataFrame([summary]).to_csv(os.path.join(OUTPUT_DIR, "summary_stats.csv"), index=False)
print("summary_stats.csv written")

label_stats = df.groupby("label").agg(
    count=("score", "size"),
    mean_score=("score", "mean"),
    std_score=("score", "std"),
    median_score=("score", "median"),
    mean_span_len=("span_length", "mean"),
    pct_low_conf=("score", lambda x: (x < 0.7).mean() * 100)
).reset_index()
label_stats.to_csv(os.path.join(OUTPUT_DIR, "label_stats.csv"), index=False)
print("label_stats.csv written")

fold_stats = df.groupby(["fold", "label"]).agg(
    count=("score", "size"),
    mean_score=("score", "mean"),
).reset_index()
fold_stats.to_csv(os.path.join(OUTPUT_DIR, "fold_stats.csv"), index=False)
print("fold_stats.csv written")

top10_per_label = (
    df.groupby(["label", "span_normalized"])
    .size()
    .reset_index(name="count")
    .sort_values(["label", "count"], ascending=[True, False])
    .groupby("label")
    .head(10)
    .reset_index(drop=True)
)
top10_per_label.to_csv(os.path.join(OUTPUT_DIR, "top10_mentions_per_label.csv"), index=False)
print("top10_mentions_per_label.csv written")

total_entities = len(df)
label_frequency = (
    df.groupby("label")
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
    .reset_index(drop=True)
)
label_frequency["pct_total"] = (label_frequency["count"] / total_entities * 100).round(2)
label_frequency.to_csv(os.path.join(OUTPUT_DIR, "label_frequency_table.csv"), index=False)
print("label_frequency_table.csv written")

fig1 = px.bar(
    label_stats.sort_values("count", ascending=False),
    x="label", y="count",
    title="Entity count by label type",
)
fig1.update_xaxes(title_text="Entity label")
fig1.update_yaxes(title_text="Count")
fig1.update_traces(cliponaxis=False)
fig1.write_html(os.path.join(OUTPUT_DIR, "chart_entity_frequency.html"))
print("chart_entity_frequency.html written")

fig2 = px.box(df, x="label", y="score", title="Confidence score distribution by label")
fig2.update_xaxes(title_text="Entity label")
fig2.update_yaxes(title_text="Conf. score")
fig2.write_html(os.path.join(OUTPUT_DIR, "chart_score_by_label.html"))
print("chart_score_by_label.html written")

thresholds = np.arange(0.0, 1.01, 0.01)
coverage = [(df["score"] >= t).mean() * 100 for t in thresholds]
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=thresholds, y=coverage, mode="lines", fill="tozeroy", name="Coverage"))
fig3.update_layout(title="Entity retention vs confidence threshold")
fig3.update_xaxes(title_text="Score threshold")
fig3.update_yaxes(title_text="Entities retained %")
fig3.write_html(os.path.join(OUTPUT_DIR, "chart_coverage_threshold.html"))
print("chart_coverage_threshold.html written")

ents_per_patient = df.groupby("patient_id").size().reset_index(name="n_entities")
fig4 = px.histogram(ents_per_patient, x="n_entities", nbins=40, title="Entities per patient distribution")
fig4.update_xaxes(title_text="Entities/patient")
fig4.update_yaxes(title_text="Nº patients")
fig4.write_html(os.path.join(OUTPUT_DIR, "chart_entities_per_patient.html"))
print("chart_entities_per_patient.html written")

fig5 = px.line(
    fold_stats, x="fold", y="mean_score", color="label",
    markers=True, title="Mean score per fold by entity label"
)
fig5.update_xaxes(title_text="Fold")
fig5.update_yaxes(title_text="Mean score")
fig5.update_layout(
    legend=dict(
        orientation='v',
        yanchor='middle',
        y=0.5,
        xanchor='left',
        x=1.02
    ),
    margin=dict(r=200)
)
fig5.write_html(os.path.join(OUTPUT_DIR, "chart_fold_consistency.html"))

fig6 = px.box(df, x="label", y="span_length", title="Span length by entity label")
fig6.update_xaxes(title_text="Entity label")
fig6.update_yaxes(title_text="Span length (chars)")
fig6.write_html(os.path.join(OUTPUT_DIR, "chart_span_length.html"))
print("chart_span_length.html written")

print("Done. All files written to:", OUTPUT_DIR)
