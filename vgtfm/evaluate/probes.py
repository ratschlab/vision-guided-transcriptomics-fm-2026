"""Annotation-prediction probes and their metrics.

The probe is a 5-nearest-neighbour classifier, kept weak so that it measures what
the *embedding* knows rather than what a classifier can learn on top of it. Two
details matter for comparability:

* **Standardisation** is fitted on the probe's training rows only. Without it, kNN
  distances are dominated by whichever embedding happens to have the largest
  leading-eigenvalue spread, which systematically favours raw PCA scores over
  LayerNorm'd neural embeddings.
* **Subsampling** of the probe's training set is stratified by class, using the
  floor-plus-largest-fractional-remainder rule, so rare classes (USZ TLS has ~800
  spots in the whole cohort) are not sampled out of existence.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier


def stratified_subsample(
    *arrays: np.ndarray, labels: np.ndarray, max_samples: int | None, random_seed: int = 42
):
    """Subsample *arrays* to at most *max_samples* rows, preserving class shares.

    Each class receives ``floor(n_c * max/n)`` rows and the remaining slots go to
    the classes with the largest fractional parts, ties broken randomly.
    """
    n = len(arrays[0])
    if max_samples is None or n <= max_samples:
        return arrays

    rng = np.random.default_rng(random_seed)
    uniq, counts = np.unique(labels, return_counts=True)
    exact = counts * max_samples / n
    takes = np.floor(exact).astype(int)
    shortfall = max_samples - takes.sum()
    if shortfall > 0:
        noise = rng.random(len(exact)) * 1e-9  # random tie-breaking
        takes[np.argsort((exact - takes) + noise)[-shortfall:]] += 1

    parts = []
    for lbl, take in zip(uniq, takes):
        if take > 0:
            idx = np.where(labels == lbl)[0]
            parts.append(rng.choice(idx, size=min(take, len(idx)), replace=False))
    idx = np.sort(np.concatenate(parts))
    return tuple(a[idx] for a in arrays)


def standardize(train_X: np.ndarray, *others: np.ndarray):
    """Zero-mean/unit-variance using the training rows' statistics only."""
    mu = train_X.mean(axis=0, keepdims=True)
    sd = train_X.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return ((train_X - mu) / sd, *[(o - mu) / sd for o in others])


def knn_predict(train_X, train_y, test_X, *, k: int = 5, n_jobs: int = -1) -> np.ndarray:
    """Fit a kNN classifier on the training rows and predict the test rows."""
    k = int(min(k, len(train_X)))
    clf = KNeighborsClassifier(n_neighbors=max(k, 1), n_jobs=n_jobs)
    clf.fit(train_X, train_y)
    return clf.predict(test_X)


def majority_predict(train_y, n_test: int) -> np.ndarray:
    """Predict the most frequent training class for every test row.

    The chance floor every embedding must clear. It never looks at the features, so
    it is identical across models and substrates by construction.
    """
    uniq, counts = np.unique(train_y, return_counts=True)
    return np.full(n_test, uniq[int(np.argmax(counts))], dtype=object)


class LabelCodec:
    """Fixed class vocabulary and integer encoding.

    Encoding once and scoring on integers keeps a 2000-replicate donor bootstrap
    affordable: sklearn re-derives the label set on every call, which dominates the
    runtime when the statistic itself is a single pass over counts.
    """

    def __init__(self, classes):
        self.classes = np.asarray(sorted({str(c) for c in classes}), dtype=object)
        self._index = {c: i for i, c in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.classes)

    def encode(self, y) -> np.ndarray:
        """Map labels to ``[0, K)``; anything outside the vocabulary becomes -1."""
        return np.fromiter(
            (self._index.get(str(v), -1) for v in np.asarray(y)), dtype=np.int64, count=len(y)
        )

    def subset_index(self, classes) -> np.ndarray:
        """Positions of *classes* within the vocabulary, for macro averaging."""
        return np.array(
            sorted(self._index[str(c)] for c in classes if str(c) in self._index), dtype=np.int64
        )


def confusion_counts(true_codes: np.ndarray, pred_codes: np.ndarray, n_classes: int) -> np.ndarray:
    """``(K, K)`` confusion matrix from integer codes; rows with -1 are dropped."""
    keep = (true_codes >= 0) & (pred_codes >= 0)
    K = int(n_classes)
    return np.bincount(true_codes[keep] * K + pred_codes[keep], minlength=K * K).reshape(K, K)


def metrics_from_confusion(cm: np.ndarray, score_idx: np.ndarray | None = None) -> dict:
    """Accuracy plus per-class and macro precision/recall/F1 from a confusion matrix.

    Matches sklearn's ``precision_recall_fscore_support(labels=..., zero_division=0)``:
    the confusion matrix always spans the *full* vocabulary, so a prediction that
    falls outside the averaged subset still counts as a false negative for its true
    class, while ``score_idx`` selects which classes the macro average runs over.
    That distinction matters for the per-tissue breakdowns, where the probe can
    predict a class that does not occur in the held-out tissue.
    """
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        recall = np.where(support > 0, tp / support, 0.0)
        denom = precision + recall
        f1 = np.where(denom > 0, 2 * precision * recall / denom, 0.0)

    n = float(cm.sum())
    idx = np.arange(len(tp)) if score_idx is None else np.asarray(score_idx)
    return {
        "accuracy": float(tp.sum() / n) if n > 0 else 0.0,
        "precision": float(precision[idx].mean()) if len(idx) else 0.0,
        "recall": float(recall[idx].mean()) if len(idx) else 0.0,
        "f1_score": float(f1[idx].mean()) if len(idx) else 0.0,
        "n_test": int(n),
        "_per_class": (precision, recall, f1, support),
    }


def classification_metrics(
    y_true, y_pred, classes=None, codec: "LabelCodec | None" = None, score_classes=None
) -> dict:
    """Macro accuracy/precision/recall/F1 plus a per-class breakdown.

    ``classes`` is the full vocabulary; ``score_classes`` (default: all of it) is the
    subset the macro average runs over. Fixing the denominator matters — averaging
    over whichever classes happen to appear in a fold makes folds incomparable.
    """
    if codec is None:
        if classes is None:
            classes = np.unique(
                np.concatenate([np.asarray(y_true).astype(str), np.asarray(y_pred).astype(str)])
            )
        codec = LabelCodec(classes)

    cm = confusion_counts(codec.encode(y_true), codec.encode(y_pred), len(codec))
    score_idx = None if score_classes is None else codec.subset_index(score_classes)
    out = metrics_from_confusion(cm, score_idx)
    precision, recall, f1, support = out.pop("_per_class")
    out["per_class"] = {
        str(c): {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1_score": float(f1[i]),
            "support": int(support[i]),
        }
        for i, c in enumerate(codec.classes)
    }
    return out


def sklearn_reference_metrics(y_true, y_pred, classes) -> dict:
    """Same statistics via sklearn. Used only to unit-test the fast path."""
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)
    classes = np.asarray(sorted({str(c) for c in classes}))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, average="macro", zero_division=0
    )
    pc_p, pc_r, pc_f, pc_s = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, average=None, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "per_class": {
            str(c): {
                "precision": float(pc_p[i]),
                "recall": float(pc_r[i]),
                "f1_score": float(pc_f[i]),
                "support": int(pc_s[i]),
            }
            for i, c in enumerate(classes)
        },
    }
