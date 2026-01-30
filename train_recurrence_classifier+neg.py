import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          AutoModelForTokenClassification, TrainingArguments,
                          Trainer, pipeline)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report, \
    roc_curve, roc_auc_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# Config
TRAIN_CSV = "train.csv"
VAL_CSV = "validation.csv"
TEST_CSV = "test.csv"
MODEL_NAME = "XLM-RoBERTa-Base"
NER_NEGATION_MODEL = "medspaner/roberta-es-clinical-trials-neg-spec-ner"
OUTPUT_DIR = "recurrence_classifier_RobertaNEW"
MAX_LENGTH = 512
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 2e-5

CONFIDENCE_THRESHOLD = 0.8
NEGATION_SCOPE_WINDOW = 50

os.makedirs(OUTPUT_DIR, exist_ok=True)


print(f"Config:")
print(f" Estrategia 1: Scope de negación (±{NEGATION_SCOPE_WINDOW} chars)")
print(f" Estrategia 2: Umbral de confianza = {CONFIDENCE_THRESHOLD}")
print(f" Lógica: Solo va a corrigir si (negación afecta recurrencia) y la confianza < {CONFIDENCE_THRESHOLD}")


print("\nCargando datos...")
train_df = pd.read_csv(TRAIN_CSV, encoding='utf-8')
val_df = pd.read_csv(VAL_CSV, encoding='utf-8')
test_df = pd.read_csv(TEST_CSV, encoding='utf-8')

print(f"Train: {len(train_df)} (Pos: {train_df['recurrencia'].sum()})")
print(f"Val: {len(val_df)} (Pos: {val_df['recurrencia'].sum()})")
print(f"Test: {len(test_df)} (Pos: {test_df['recurrencia'].sum()})")


train_pos = train_df[train_df['recurrencia'] == 1].copy()
train_neg = train_df[train_df['recurrencia'] == 0].copy()
extra_pos = train_pos.sample(n=int(len(train_pos) * 0.35), replace=True, random_state=42)
train_augmented = pd.concat([train_neg, train_pos, extra_pos], ignore_index=True)
train_augmented = train_augmented.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Train tras oversample: {len(train_augmented)}")
print(f"  Positivos: {train_augmented['recurrencia'].sum()}")

class_weights = np.array([.98, 1.25])
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
print(f"\nPesos de clase: {class_weights}")


class RecurrenceDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row['texto_limpio'])
        label = int(row['recurrencia'])

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


print(f"\nCargando modelo: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

train_dataset = RecurrenceDataset(train_augmented, tokenizer, MAX_LENGTH)
val_dataset = RecurrenceDataset(val_df, tokenizer, MAX_LENGTH)
test_dataset = RecurrenceDataset(test_df, tokenizer, MAX_LENGTH)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}


training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    warmup_ratio=0.1,
    weight_decay=0.05,
    logging_steps=20,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    save_total_limit=2,
    report_to="none",
    gradient_accumulation_steps=4,
    fp16=torch.cuda.is_available()
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    class_weights=class_weights_tensor
)

print("\nEntrenando")
trainer.train()

print("\nEvaluando test")
test_results = trainer.predict(test_dataset)

probs = torch.softmax(torch.tensor(test_results.predictions), dim=1)[:, 1].numpy()
labels = test_results.label_ids
fpr, tpr, thresholds = roc_curve(labels, probs)
f1_scores_list = []
for thresh in thresholds:
    preds_t = (probs >= thresh).astype(int)
    _, _, f1, _ = precision_recall_fscore_support(labels, preds_t, average='binary', zero_division=0)
    f1_scores_list.append(f1)

best_threshold = thresholds[np.argmax(f1_scores_list)]
preds_opt = (probs >= best_threshold).astype(int)

print(f"\nThreshold optimo: {best_threshold:.3f}")
print("\nResultados SIN corrección por negación:")
print(classification_report(labels, preds_opt, target_names=['No Recurrencia', 'Recurrencia'], digits=3))

cm_before = confusion_matrix(labels, preds_opt)
tn_before, fp_before, fn_before, tp_before = cm_before.ravel()

print(f"\nMatriz confusion (ANTES de NER negación):")
print(f"  TN: {tn_before}, FP: {fp_before}")
print(f"  FN: {fn_before}, TP: {tp_before}")
print(f"\nCargando modelo NER: {NER_NEGATION_MODEL}")
try:
    ner_pipeline = pipeline(
        "ner",
        model=NER_NEGATION_MODEL,
        tokenizer=NER_NEGATION_MODEL,
        aggregation_strategy="simple",
        device=0 if torch.cuda.is_available() else -1
    )
    print("Modelo NER cargado correctamente")
except Exception as e:
    print(f"Error cargando modelo NER: {e}")
    ner_pipeline = None


def detect_negation_ner(text, ner_pipeline):
    """
    Detecta negaciones usando el modelo NER
    Devuelve: (has_negation, negation_entities, entity_count)
    """
    if not text or pd.isna(text) or len(str(text).strip()) < 5:
        return False, [], 0

    if ner_pipeline is None:
        return False, [], 0

    try:
        # Aplicar NER
        entities = ner_pipeline(str(text))
        # Filtrar entidades de negación (labels que contengan "NEG" o "NEGATION")
        negation_entities = [
            ent for ent in entities
            if 'NEG' in ent.get('entity_group', '').upper() or
               'NEGATION' in ent.get('entity_group', '').upper() or
               'NEGATED' in ent.get('entity_group', '').upper()
        ]

        has_negation = len(negation_entities) > 0

        return has_negation, negation_entities, len(negation_entities)

    except Exception as e:
        print(f"Error en NER para texto: {str(text)[:50]}... Error: {e}")
        return False, [], 0



def is_negation_affecting_recurrence(span, negation_entities, window=NEGATION_SCOPE_WINDOW):
    """
    Verificar si la negación detectada afecta realmente a keywords de recurrencia
    Busca keywords de recurrencia dentro de una ventana de carácteres de cada negación
    """
    if not negation_entities:
        return False

    span_lower = span.lower()

    recurrence_keywords = [
        'progresión', 'progresion', 'avance', 'empeoramiento', 'evolución desfavorable',
        'recurrencia', 'recidiva', 'recaída', 'recaida', 'reaparición', 'reaparicion',
        'nueva lesión', 'nuevo nódulo', 'nueva masa', 'nuevas lesiones', 'nuevos nódulos',
        'metástasis', 'metastasis', 'mts', 'diseminación', 'diseminacion',
        'deterioro', 'incremento', 'aumento', 'crecimiento',
        'mayor tamaño', 'más grande', 'crece', 'crecido', 'aumentado',
        'nueva', 'nuevo', 'nuevas', 'nuevos',
        'lesión', 'lesion', 'nódulo', 'nodulo', 'masa'
    ]

    for neg_ent in negation_entities:
        neg_start = neg_ent.get('start', 0)
        neg_end = neg_ent.get('end', 0)
        window_start = max(0, neg_start - window)
        window_end = min(len(span), neg_end + window)
        window_text = span_lower[window_start:window_end]
        if any(keyword in window_text for keyword in recurrence_keywords):
            return True

    return False



test_df['predicted_original'] = preds_opt
test_df['predicted_proba'] = probs
negation_results = []
for idx, row in test_df.iterrows():
    span = row.get('span_ampliado', '')
    has_neg, neg_entities, neg_count = detect_negation_ner(span, ner_pipeline)

    negation_affects_recurrence = False

    if has_neg and neg_entities:
        negation_affects_recurrence = is_negation_affecting_recurrence(span, neg_entities)

    negation_results.append({
        'has_negation_ner': has_neg,
        'negation_count': neg_count,
        'negation_entities': neg_entities,
        'negation_affects_recurrence': negation_affects_recurrence  # ESTRATEGIA 1
    })

    if idx % 20 == 0:
        print(f"  Procesado: {idx + 1}/{len(test_df)}", end='\r')

print(f"  Procesado: {len(test_df)}/{len(test_df)} ")

# Añadir resultados al dataframe
test_df['has_negation_ner'] = [r['has_negation_ner'] for r in negation_results]
test_df['negation_count'] = [r['negation_count'] for r in negation_results]
test_df['negation_affects_recurrence'] = [r['negation_affects_recurrence'] for r in negation_results]
test_df['predicted_corrected'] = preds_opt.copy()
test_df['correction_reason'] = 'Sin corrección'

corrected_count = 0
skipped_no_scope = 0
skipped_high_confidence = 0
skipped_both = 0

for idx in range(len(test_df)):
    proba = test_df.iloc[idx]['predicted_proba']
    has_negation = test_df.iloc[idx]['has_negation_ner']
    affects_recurrence = test_df.iloc[idx]['negation_affects_recurrence']
    original_pred = preds_opt[idx]

    if original_pred == 1 and has_negation:

        scope_condition = affects_recurrence
        confidence_condition = proba < CONFIDENCE_THRESHOLD

        if scope_condition and confidence_condition:
            # Ambas condiciones se cumplen → CORREGIR
            test_df.at[idx, 'predicted_corrected'] = 0
            test_df.at[idx, 'correction_reason'] = f'Negación en scope + Baja confianza ({proba:.3f})'
            corrected_count += 1

        elif not scope_condition and not confidence_condition:
            # Ninguna condición se cumple
            test_df.at[idx, 'correction_reason'] = f'Negación fuera de scope + Alta confianza ({proba:.3f})'
            skipped_both += 1

        elif not scope_condition:
            # Solo falla scope
            test_df.at[idx, 'correction_reason'] = f'Negación fuera de scope (no afecta recurrencia)'
            skipped_no_scope += 1

        elif not confidence_condition:
            # Solo falla confidence
            test_df.at[idx, 'correction_reason'] = f'Negación en scope pero Alta confianza ({proba:.3f})'
            skipped_high_confidence += 1

print(f"\nResultados de corrección híbrida:")
print(f"Predicciones corregidas: {corrected_count}")
print(f"corregidos por scope (negación no afecta recurrencia): {skipped_no_scope}")
print(f" corregidos por confianza (probabilidad ≥ {CONFIDENCE_THRESHOLD}): {skipped_high_confidence}")
print(f" corregidos por ambos: {skipped_both}")

# Estadísticas de negación
total_with_negation = test_df['has_negation_ner'].sum()
total_with_relevant_negation = test_df['negation_affects_recurrence'].sum()

print(f"\n Total casos con negación NER: {total_with_negation} ({total_with_negation / len(test_df) * 100:.1f}%)")
print(
    f" Negación afecta recurrencia: {total_with_relevant_negation}/{total_with_negation} ({total_with_relevant_negation / total_with_negation * 100 if total_with_negation > 0 else 0:.1f}%)")
print(
    f" Tasa de corrección: {corrected_count}/{total_with_negation} ({corrected_count / total_with_negation * 100 if total_with_negation > 0 else 0:.1f}%)")

# Crea DataFrame con toda la información
detailed_analysis = test_df.copy()

# Añade la información de negación detallada
detailed_analysis['negation_entities_text'] = ''
for idx, result in enumerate(negation_results):
    if result['negation_entities']:
        # Extraer texto de las entidades de negación
        entity_texts = [
            f"{ent.get('word', '')} ({ent.get('entity_group', '')}:{ent.get('score', 0):.2f})"
            for ent in result['negation_entities']
        ]
        detailed_analysis.at[idx, 'negation_entities_text'] = ' | '.join(entity_texts)

# Marcar que cambios ha habido
detailed_analysis['prediction_changed'] = (
        detailed_analysis['predicted_original'] != detailed_analysis['predicted_corrected']
)


# Clasifica el tipo de cambio
def classify_change(row):
    if not row['prediction_changed']:
        return 'Sin cambio'

    original = row['predicted_original']
    corrected = row['predicted_corrected']
    actual = row['recurrencia']

    if original == 1 and corrected == 0:
        if actual == 0:
            return 'FP → TN (Mejora )'
        else:
            return 'TP → FN (Empeora ⚠)'

    return 'Otro'


detailed_analysis['change_type'] = detailed_analysis.apply(classify_change, axis=1)

# Clasifica entre antes y después
def classify_correctness_before(row):
    pred = row['predicted_original']
    actual = row['recurrencia']
    if pred == actual:
        return 'TP' if pred == 1 else 'TN'
    else:
        return 'FP' if pred == 1 else 'FN'


def classify_correctness_after(row):
    pred = row['predicted_corrected']
    actual = row['recurrencia']
    if pred == actual:
        return 'TP' if pred == 1 else 'TN'
    else:
        return 'FP' if pred == 1 else 'FN'


detailed_analysis['classification_before'] = detailed_analysis.apply(classify_correctness_before, axis=1)
detailed_analysis['classification_after'] = detailed_analysis.apply(classify_correctness_after, axis=1)

# Reorganiza columnas para mejor visualización y comparar
columns_order = [
    'recurrencia',
    'predicted_original',
    'predicted_corrected',
    'predicted_proba',
    'prediction_changed',
    'change_type',
    'correction_reason',
    'classification_before',
    'classification_after',
    'has_negation_ner',
    'negation_count',
    'negation_affects_recurrence',  # ESTRATEGIA 1
    'negation_entities_text',
    'span_ampliado',
    'texto_limpio'
]
final_columns = [col for col in columns_order if col in detailed_analysis.columns]
for col in detailed_analysis.columns:
    if col not in final_columns:
        final_columns.append(col)

detailed_analysis = detailed_analysis[final_columns]
detailed_analysis.to_csv(f'{OUTPUT_DIR}/detailed_negation_analysis.csv', index=False, encoding='utf-8')
print(f" Análisis detallado guardado: {OUTPUT_DIR}/detailed_negation_analysis.csv")

changed_cases = detailed_analysis[detailed_analysis['prediction_changed']].copy()
changed_cases.to_csv(f'{OUTPUT_DIR}/cases_changed_by_ner.csv', index=False, encoding='utf-8')
improvements = detailed_analysis[detailed_analysis['change_type'] == 'FP → TN'].copy()
improvements.to_csv(f'{OUTPUT_DIR}/improvements_fp_to_tn.csv', index=False, encoding='utf-8')
worsenings = detailed_analysis[detailed_analysis['change_type'] == 'TP → FN'].copy()
worsenings.to_csv(f'{OUTPUT_DIR}/worsenings_tp_to_fn.csv', index=False, encoding='utf-8')
with_negation = detailed_analysis[detailed_analysis['has_negation_ner']].copy()
with_negation.to_csv(f'{OUTPUT_DIR}/all_cases_with_negation.csv', index=False, encoding='utf-8')
relevant_negation = detailed_analysis[detailed_analysis['negation_affects_recurrence']].copy()
relevant_negation.to_csv(f'{OUTPUT_DIR}/negation_affects_recurrence.csv', index=False, encoding='utf-8')

fp_not_corrected = detailed_analysis[
    (detailed_analysis['classification_before'] == 'FP') &
    (detailed_analysis['classification_after'] == 'FP')
    ].copy()
fp_not_corrected.to_csv(f'{OUTPUT_DIR}/fp_not_corrected.csv', index=False, encoding='utf-8')

not_corrected_scope = detailed_analysis[
    (detailed_analysis['has_negation_ner']) &
    (~detailed_analysis['negation_affects_recurrence']) &
    (detailed_analysis['predicted_original'] == 1)
    ].copy()
not_corrected_scope.to_csv(f'{OUTPUT_DIR}/not_corrected_no_scope.csv', index=False, encoding='utf-8')
print(f" (fuera de scope): {OUTPUT_DIR}/not_corrected_no_scope.csv ({len(not_corrected_scope)} casos)")

not_corrected_confidence = detailed_analysis[
    (detailed_analysis['has_negation_ner']) &
    (detailed_analysis['negation_affects_recurrence']) &
    (~detailed_analysis['prediction_changed']) &
    (detailed_analysis['predicted_original'] == 1)
    ].copy()
not_corrected_confidence.to_csv(f'{OUTPUT_DIR}/not_corrected_high_confidence.csv', index=False, encoding='utf-8')
print(
    f" sin corregir pro (alta confianza): {OUTPUT_DIR}/not_corrected_high_confidence.csv ({len(not_corrected_confidence)} casos)")


summary_stats = {
    'strategy': 'Hibrida',
    'confidence_threshold': float(CONFIDENCE_THRESHOLD),
    'scope_window': int(NEGATION_SCOPE_WINDOW),
    'total_cases': len(detailed_analysis),
    'cases_with_negation': int(detailed_analysis['has_negation_ner'].sum()),
    'cases_negation_affects_recurrence': int(detailed_analysis['negation_affects_recurrence'].sum()),
    'predictions_changed': int(detailed_analysis['prediction_changed'].sum()),
    'skipped_no_scope': int(skipped_no_scope),
    'skipped_high_confidence': int(skipped_high_confidence),
    'skipped_both': int(skipped_both),
    'improvements_fp_to_tn': len(improvements),
    'worsenings_tp_to_fn': len(worsenings),
    'fp_before': int((detailed_analysis['classification_before'] == 'FP').sum()),
    'fp_after': int((detailed_analysis['classification_after'] == 'FP').sum()),
    'fn_before': int((detailed_analysis['classification_before'] == 'FN').sum()),
    'fn_after': int((detailed_analysis['classification_after'] == 'FN').sum()),
    'tp_before': int((detailed_analysis['classification_before'] == 'TP').sum()),
    'tp_after': int((detailed_analysis['classification_after'] == 'TP').sum()),
    'tn_before': int((detailed_analysis['classification_before'] == 'TN').sum()),
    'tn_after': int((detailed_analysis['classification_after'] == 'TN').sum()),
}

# Guardar resumen
with open(f'{OUTPUT_DIR}/negation_changes_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary_stats, f, indent=2, ensure_ascii=False)
print(f" Resumen guardado: {OUTPUT_DIR}/negation_changes_summary.json")

print(f"""
ESTADÍSTICAS:
  Total casos: {summary_stats['total_cases']}
  Con negación detectada: {summary_stats['cases_with_negation']} ({summary_stats['cases_with_negation'] / summary_stats['total_cases'] * 100:.1f}%)
  Negación afecta recurrencia: {summary_stats['cases_negation_affects_recurrence']}/{summary_stats['cases_with_negation']} ({summary_stats['cases_negation_affects_recurrence'] / summary_stats['cases_with_negation'] * 100 if summary_stats['cases_with_negation'] > 0 else 0:.1f}%)
  Predicciones cambiadas: {summary_stats['predictions_changed']}

NO CORREGIDOS:
  Fuera de scope: {summary_stats['skipped_no_scope']}
  Alta confianza: {summary_stats['skipped_high_confidence']}
  Ambos: {summary_stats['skipped_both']}

CAMBIOS:
  Mejoras (FP→TN): {summary_stats['improvements_fp_to_tn']}
  Empeoramientos (TP→FN): {summary_stats['worsenings_tp_to_fn']}

ANTES NER:
  TP: {summary_stats['tp_before']}, TN: {summary_stats['tn_before']}
  FP: {summary_stats['fp_before']}, FN: {summary_stats['fn_before']}

DESPUÉS NER:
  TP: {summary_stats['tp_after']}, TN: {summary_stats['tn_after']}
  FP: {summary_stats['fp_after']}, FN: {summary_stats['fn_after']}
""")


print("Ejemplos dif casos")


if len(improvements) > 0:
    print("\n MEJORAS (FP → TN):")
    print("-" * 80)
    for idx, (i, row) in enumerate(improvements.head(5).iterrows(), 1):
        print(
            f"\n{idx}. Prob: {row['predicted_proba']:.3f} | Negaciones: {row['negation_count']} | Scope: {row['negation_affects_recurrence']}")
        print(f"  Razón: {row['correction_reason']}")
        print(f"  Entidades: {row['negation_entities_text'][:100]}")
        print(f"  Span: {str(row['span_ampliado'])[:150]}...")

if len(worsenings) > 0:
    print("Los que empeoran: ")
    print("-" * 80)
    for idx, (i, row) in enumerate(worsenings.head(5).iterrows(), 1):
        print(
            f"\n{idx}. Prob: {row['predicted_proba']:.3f} | Negaciones: {row['negation_count']} | Scope: {row['negation_affects_recurrence']}")
        print(f"   Razón: {row['correction_reason']}")
        print(f"   Entidades: {row['negation_entities_text'][:100]}")
        print(f"   Span: {str(row['span_ampliado'])[:150]}...")

if len(not_corrected_scope) > 0:
    print("\n NO CORREGIDOS (Negación fuera de scope):")
    for idx, (i, row) in enumerate(not_corrected_scope.head(3).iterrows(), 1):
        print(
            f"\n{idx}. Prob: {row['predicted_proba']:.3f} | Real: {row['recurrencia']} ({'TP' if row['recurrencia'] == 1 else 'FP'})")
        print(f"   Razón: {row['correction_reason']}")
        print(f"   Entidades: {row['negation_entities_text'][:100]}")
        print(f"   Span: {str(row['span_ampliado'])[:150]}...")

if len(not_corrected_confidence) > 0:
    print("\n NO CORREGIDOS (Alta confianza):")
    print("-" * 80)
    for idx, (i, row) in enumerate(not_corrected_confidence.head(3).iterrows(), 1):
        print(
            f"\n{idx}. Prob: {row['predicted_proba']:.3f} | Real: {row['recurrencia']} ({'TP' if row['recurrencia'] == 1 else 'FP'})")
        print(f"   Razón: {row['correction_reason']}")
        print(f"   Entidades: {row['negation_entities_text'][:100]}")
        print(f"   Span: {str(row['span_ampliado'])[:150]}...")

print("Correcciones")

preds_corrected = test_df['predicted_corrected'].values

print("\nResultados CON corrección por NER:")
print(classification_report(labels, preds_corrected, target_names=['No Recurrencia', 'Recurrencia'], digits=3))

# Matriz de confusion DESPUÉS de negación
cm_after = confusion_matrix(labels, preds_corrected)
tn_after, fp_after, fn_after, tp_after = cm_after.ravel()

print(f"\nMatriz confusion (DESPUÉS de NER):")
print(f"  TN: {tn_after}, FP: {fp_after}")
print(f"  FN: {fn_after}, TP: {tp_after}")

# Calcular métricas antes
precision_before = precision_recall_fscore_support(labels, preds_opt, average='binary')[0]
recall_before = precision_recall_fscore_support(labels, preds_opt, average='binary')[1]
f1_before = f1_score(labels, preds_opt)

# Calcular métricas después
precision_after = tp_after / (tp_after + fp_after) if (tp_after + fp_after) > 0 else 0
recall_after = tp_after / (tp_after + fn_after) if (tp_after + fn_after) > 0 else 0
f1_after = 2 * precision_after * recall_after / (precision_after + recall_after) if (precision_after + recall_after) > 0 else 0
accuracy_after = (tn_after + tp_after) / (tn_after + fp_after + fn_after + tp_after)
specificity_after = tn_after / (tn_after + fp_after) if (tn_after + fp_after) > 0 else 0
sensitivity_after = tp_after / (tp_after + fn_after) if (tp_after + fn_after) > 0 else 0
auc_score = roc_auc_score(labels, probs)

# Comparación
fp_reduction = fp_before - fp_after
fn_increase = fn_after - fn_before

print(f"""
MÉTRICA              ANTES    DESPUÉS   CAMBIO
{'─' * 50}
TN (Verdaderos Neg)  {tn_before:3d}      {tn_after:3d}       {tn_after - tn_before:+3d}
FP (Falsos Pos)      {fp_before:3d}      {fp_after:3d}       {fp_after - fp_before:+3d} {'' if fp_after < fp_before else '⚠'}
FN (Falsos Neg)      {fn_before:3d}      {fn_after:3d}       {fn_after - fn_before:+3d} {'⚠' if fn_after > fn_before else ''}
TP (Verdaderos Pos)  {tp_before:3d}      {tp_after:3d}       {tp_after - tp_before:+3d}

Precision            {precision_before:.3f}    {precision_after:.3f}     {precision_after - precision_before:+.3f}
Recall               {recall_before:.3f}    {recall_after:.3f}     {recall_after - recall_before:+.3f}
F1-Score             {f1_before:.3f}    {f1_after:.3f}     {f1_after - f1_before:+.3f} {'' if f1_after > f1_before else '⚠'}

Reducción FP: {fp_reduction} casos ({fp_reduction / fp_before * 100 if fp_before > 0 else 0:.1f}%)
Aumento FN: {fn_increase} casos ({fn_increase / fn_before * 100 if fn_before > 0 else 0:.1f}%)

Balance Neto: {(tn_after - tn_before) - (fn_after - fn_before)} casos mejor clasificados
""")

# Metricas por clase
precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
    labels, preds_corrected, average=None
)

# Guardar resultados en un JSON
results = {
    "model_info": {
        "model_name": MODEL_NAME,
        "ner_model": NER_NEGATION_MODEL,
        "model_dir": OUTPUT_DIR,
        "threshold": float(best_threshold),
        "total_samples": int(len(labels)),
        "strategy": "Híbrida",
        "confidence_threshold": float(CONFIDENCE_THRESHOLD),
        "scope_window": int(NEGATION_SCOPE_WINDOW)
    },
    "before_ner_correction": {
        "confusion_matrix": {
            "true_negative": int(tn_before),
            "false_positive": int(fp_before),
            "false_negative": int(fn_before),
            "true_positive": int(tp_before)
        },
        "metrics": {
            "precision": float(precision_before),
            "recall": float(recall_before),
            "f1_score": float(f1_before)
        }
    },
    "after_ner_correction": {
        "confusion_matrix": {
            "true_negative": int(tn_after),
            "false_positive": int(fp_after),
            "false_negative": int(fn_after),
            "true_positive": int(tp_after)
        },
        "metrics_overall": {
            "accuracy": float(accuracy_after),
            "precision": float(precision_after),
            "recall": float(recall_after),
            "f1_score": float(f1_after),
            "sensitivity": float(sensitivity_after),
            "specificity": float(specificity_after),
            "auc_roc": float(auc_score)
        },
        "metrics_per_class": {
            "class_0_no_recurrence": {
                "precision": float(precision_per_class[0]),
                "recall": float(recall_per_class[0]),
                "f1_score": float(f1_per_class[0]),
                "support": int(support_per_class[0])
            },
            "class_1_recurrence": {
                "precision": float(precision_per_class[1]),
                "recall": float(recall_per_class[1]),
                "f1_score": float(f1_per_class[1]),
                "support": int(support_per_class[1])
            }
        }
    },
    "ner_correction_stats": {
        "predictions_corrected": int(corrected_count),
        "total_with_negation": int(total_with_negation),
        "negation_affects_recurrence": int(total_with_relevant_negation),
        "skipped_no_scope": int(skipped_no_scope),
        "skipped_high_confidence": int(skipped_high_confidence),
        "skipped_both": int(skipped_both),
        "fp_reduction": int(fp_reduction),
        "fn_increase": int(fn_increase),
        "net_improvement": int((tn_after - tn_before) - (fn_after - fn_before))
    }
}

json_path = f'{OUTPUT_DIR}/evaluation_results.json'
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nResultados JSON guardados: {json_path}")


# 1. Matriz de confusion comparativa
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
# Antes - absoluta
sns.heatmap(cm_before, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
            xticklabels=['No Recurrencia', 'Recurrencia'],
            yticklabels=['No Recurrencia', 'Recurrencia'],
            cbar_kws={'label': 'Count'})
axes[0, 0].set_xlabel('Predicho', fontsize=12)
axes[0, 0].set_ylabel('Real', fontsize=12)
axes[0, 0].set_title(f'ANTES NER\n(Threshold: {best_threshold:.3f})', fontsize=14, fontweight='bold')

# Antes - normalizada
cm_before_norm = cm_before.astype('float') / cm_before.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_before_norm, annot=True, fmt='.2%', cmap='Greens', ax=axes[0, 1],
            xticklabels=['No Recurrencia', 'Recurrencia'],
            yticklabels=['No Recurrencia', 'Recurrencia'],
            cbar_kws={'label': 'Proportion'})
axes[0, 1].set_xlabel('Predicho', fontsize=12)
axes[0, 1].set_ylabel('Real', fontsize=12)
axes[0, 1].set_title('ANTES NER (Normalizada)', fontsize=14, fontweight='bold')

# Después - absoluta
sns.heatmap(cm_after, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
            xticklabels=['No Recurrencia', 'Recurrencia'],
            yticklabels=['No Recurrencia', 'Recurrencia'],
            cbar_kws={'label': 'Count'})
axes[1, 0].set_xlabel('Predicho', fontsize=12)
axes[1, 0].set_ylabel('Real', fontsize=12)
axes[1, 0].set_title(
    f'DESPUÉS NER \nFP: {fp_before}→{fp_after} (-{fp_reduction}) | FN: {fn_before}→{fn_after} (+{fn_increase})',
    fontsize=14, fontweight='bold')

# Después - normalizada
cm_after_norm = cm_after.astype('float') / cm_after.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_after_norm, annot=True, fmt='.2%', cmap='Greens', ax=axes[1, 1],
            xticklabels=['No Recurrencia', 'Recurrencia'],
            yticklabels=['No Recurrencia', 'Recurrencia'],
            cbar_kws={'label': 'Proportion'})
axes[1, 1].set_xlabel('Predicho', fontsize=12)
axes[1, 1].set_ylabel('Real', fontsize=12)
axes[1, 1].set_title(f'DESPUÉS NER (Normalizada)\nScope ±{NEGATION_SCOPE_WINDOW} | Conf < {CONFIDENCE_THRESHOLD}',
                     fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/confusion_matrix_comparison.png', dpi=300, bbox_inches='tight')
print(f" Matriz comparativa guardada: {OUTPUT_DIR}/confusion_matrix_comparison.png")
plt.close()

# 2. Métricas comparativas
fig, ax = plt.subplots(figsize=(12, 6))

metrics_names = ['Precision', 'Recall', 'F1-Score', 'Specificity', 'Sensitivity']
metrics_before_list = [
    precision_before,
    recall_before,
    f1_before,
    tn_before / (tn_before + fp_before),
    tp_before / (tp_before + fn_before)
]
metrics_after_list = [precision_after, recall_after, f1_after, specificity_after, sensitivity_after]

x = np.arange(len(metrics_names))
width = 0.35

bars1 = ax.bar(x - width / 2, metrics_before_list, width, label='Antes NER', color='#3498db', edgecolor='black')
bars2 = ax.bar(x + width / 2, metrics_after_list, width, label='Después NER', color='#2ecc71',
               edgecolor='black')

ax.set_xlabel('Métrica', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title(
    f'Comparación de Métricas: Antes vs Después de NER\n(Scope ±{NEGATION_SCOPE_WINDOW} chars + Confidence < {CONFIDENCE_THRESHOLD})',
    fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics_names)
ax.legend()
ax.set_ylim(0, 1.1)
ax.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/metrics_comparison_ner.png', dpi=300, bbox_inches='tight')
print(f" Comparación métricas guardada: {OUTPUT_DIR}/metrics_comparison_ner.png")
plt.close()

# 3. ROC Curve
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
ax.legend(loc="lower right", fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/roc_curve.png', dpi=300, bbox_inches='tight')
print(f" ROC curve guardada: {OUTPUT_DIR}/roc_curve.png")
plt.close()
fig, ax = plt.subplots(figsize=(14, 8))

# Datos para el flujo
stages = ['Predicciones\nPositivas', 'Con\nNegación', 'Negación\nen Scope', 'Baja\nConfianza', 'Corregidas']
positive_preds = (preds_opt == 1).sum()
with_neg = ((preds_opt == 1) & test_df['has_negation_ner']).sum()
in_scope = ((preds_opt == 1) & test_df['negation_affects_recurrence']).sum()
low_conf = ((preds_opt == 1) & test_df['negation_affects_recurrence'] & (probs < CONFIDENCE_THRESHOLD)).sum()
corrected_final = corrected_count

values = [positive_preds, with_neg, in_scope, low_conf, corrected_final]

x_pos = np.arange(len(stages))
colors_flow = ['#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#2ecc71']
bars = ax.bar(x_pos, values, color=colors_flow, edgecolor='black', linewidth=2, alpha=0.8)

# Añadir valores y porcentajes
for i, (bar, val) in enumerate(zip(bars, values)):
    if i == 0:
        pct_text = f'{val}\n(100%)'
    else:
        pct = (val / positive_preds * 100) if positive_preds > 0 else 0
        pct_text = f'{val}\n({pct:.1f}%)'

    ax.text(bar.get_x() + bar.get_width() / 2., val + 1,
            pct_text, ha='center', va='bottom', fontsize=11, fontweight='bold')

# Añadir flechas entre barras
for i in range(len(stages) - 1):
    ax.annotate('', xy=(x_pos[i + 1], values[i + 1]), xytext=(x_pos[i], values[i]),
                arrowprops=dict(arrowstyle='->', lw=2, color='black', alpha=0.5))

ax.set_xticks(x_pos)
ax.set_xticklabels(stages, fontsize=11)
ax.set_ylabel('Número de Casos', fontsize=12)
ax.set_title(
    f'Flujo de Corrección Híbrida \nScope Window: ±{NEGATION_SCOPE_WINDOW} chars | Confidence Threshold: {CONFIDENCE_THRESHOLD}',
    fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, max(values) * 1.15)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/correction_flow_hybrid.png', dpi=300, bbox_inches='tight')
print(f" Flujo de corrección guardado: {OUTPUT_DIR}/correction_flow_hybrid.png")
plt.close()

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

with open(f'{OUTPUT_DIR}/best_threshold.json', 'w') as f:
    json.dump({
        'threshold': float(best_threshold),
        'confidence_threshold': float(CONFIDENCE_THRESHOLD),
        'scope_window': int(NEGATION_SCOPE_WINDOW),
        'strategy': 'Hibrida'
    }, f, indent=2)

# Guardar predicciones con información de NER
test_df.to_csv(f'{OUTPUT_DIR}/test_predictions_with_ner.csv', index=False, encoding='utf-8')
print(f" Predicciones guardadas: {OUTPUT_DIR}/test_predictions_with_ner.csv")

print(f"Todo guardado en: {OUTPUT_DIR}/")

print("Archivos que se deberían generar:")
print(f"""
CSVs (8 archivos):
1. detailed_negation_analysis.csv - Análisis completo
2. cases_changed_by_ner.csv - Solo casos cambios
3. improvements_fp_to_tn.csv - Mejoras (FP→TN)
4. worsenings_tp_to_fn.csv - Empeoramientos (TP→FN)
5. all_cases_with_negation.csv - Todos con negación detectada
6. negation_affects_recurrence.csv - Negación en scope relevante
7. fp_not_corrected.csv - FP persistentes
8. not_corrected_no_scope.csv - NO corregidos (fuera de scope)
9. not_corrected_high_confidence.csv - NO corregidos (alta confianza)
GRÁFICAS (4 archivos):
1. confusion_matrix_comparison.png - Matrices 2x2 antes/después
2. metrics_comparison_ner.png - Barras comparativas
3. roc_curve.png - Curva ROC
4. correction_flow_hybrid.png - Flujo de corrección híbrida
JSON (2 archivos):
1. evaluation_results.json - Métricas completas
2. negation_changes_summary.json - Resumen de cambios
""")


print("\nEjecución Completada")
