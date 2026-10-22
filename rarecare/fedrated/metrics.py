"""
Evaluation Metrics Computation for RareCare Bridge.
Calculates exact multi-class accuracy, precision, recall, F1, and confusion matrix.
"""

from dataclasses import dataclass
from typing import Dict, List, Any
import numpy as np
import pandas as pd

from rarecare.config import DISEASE_CLASSES, DISEASE_SHORT_NAMES


@dataclass
class EvaluationMetrics:
    """Calculated classification metrics across all classes."""
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_class_precision: Dict[int, float]
    per_class_recall: Dict[int, float]
    per_class_f1: Dict[int, float]
    support_per_class: Dict[int, int]
    confusion_matrix: np.ndarray          # Shape: (C, C) where rows=true, cols=pred
    total_samples: int

    def to_dataframe(self) -> pd.DataFrame:
        """Convert per-class metrics into a formatted comparison table."""
        rows = []
        for c, name in DISEASE_SHORT_NAMES.items():
            rows.append({
                "Class ID": c,
                "Condition": name,
                "Precision": f"{self.per_class_precision.get(c, 0.0) * 100:.1f}%",
                "Recall": f"{self.per_class_recall.get(c, 0.0) * 100:.1f}%",
                "F1-Score": f"{self.per_class_f1.get(c, 0.0) * 100:.1f}%",
                "Support": self.support_per_class.get(c, 0),
            })
        
        # Add Macro Average summary row
        rows.append({
            "Class ID": "-",
            "Condition": "Macro Average",
            "Precision": f"{self.macro_precision * 100:.1f}%",
            "Recall": f"{self.macro_recall * 100:.1f}%",
            "F1-Score": f"{self.macro_f1 * 100:.1f}%",
            "Support": self.total_samples,
        })
        return pd.DataFrame(rows)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 4) -> EvaluationMetrics:
    """
    Compute rigorous multi-class evaluation metrics without relying on external black boxes.
    Handles edge cases like division by zero safely.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    n = len(y_true)

    if n == 0:
        return EvaluationMetrics(
            accuracy=0.0,
            macro_precision=0.0,
            macro_recall=0.0,
            macro_f1=0.0,
            per_class_precision={c: 0.0 for c in range(num_classes)},
            per_class_recall={c: 0.0 for c in range(num_classes)},
            per_class_f1={c: 0.0 for c in range(num_classes)},
            support_per_class={c: 0 for c in range(num_classes)},
            confusion_matrix=np.zeros((num_classes, num_classes), dtype=np.int64),
            total_samples=0,
        )

    # Accuracy
    accuracy = float(np.mean(y_true == y_pred))

    # Confusion Matrix: rows = true, columns = pred
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    per_class_prec = {}
    per_class_rec = {}
    per_class_f1 = {}
    support = {}

    prec_list = []
    rec_list = []
    f1_list = []

    for c in range(num_classes):
        tp = float(cm[c, c])
        fp = float(np.sum(cm[:, c]) - tp)
        fn = float(np.sum(cm[c, :]) - tp)
        class_support = int(np.sum(cm[c, :]))
        support[c] = class_support

        # Precision = TP / (TP + FP)
        prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        # Recall = TP / (TP + FN)
        rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        # F1 = 2 * (P * R) / (P + R)
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        per_class_prec[c] = prec
        per_class_rec[c] = rec
        per_class_f1[c] = f1

        prec_list.append(prec)
        rec_list.append(rec)
        f1_list.append(f1)

    macro_precision = float(np.mean(prec_list))
    macro_recall = float(np.mean(rec_list))
    macro_f1 = float(np.mean(f1_list))

    return EvaluationMetrics(
        accuracy=accuracy,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        per_class_precision=per_class_prec,
        per_class_recall=per_class_rec,
        per_class_f1=per_class_f1,
        support_per_class=support,
        confusion_matrix=cm,
        total_samples=n,
    )
