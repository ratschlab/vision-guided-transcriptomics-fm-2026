"""Annotation-label conventions, shared by evaluation and plotting.

Two rules live here, and both are load-bearing.

**Which spots count as labelled.** The manual pathology annotations cover only part
of each slide, and unlabelled spots are excluded from every label-dependent metric.
Dropping that mask changes rare-class F1 by a factor of ~3 (USZ tumour 0.51 -> 0.18).

**Which labels are the same class.** The cohorts were annotated independently and
do not share a vocabulary: TuPro writes ``Tumor`` where USZ writes ``TUM``. Every
kNN fit is within one tissue, so the split never confuses a probe, but it does
inflate the pooled macro average from 9 concepts to 10 buckets and gives the scIB
panel a label key in which one class appears twice. Names are therefore
canonicalised once, at cache-build time, with the cohort's own spelling kept in
``annotation_raw``.

:data:`CANONICAL` is minimal. Only ``TUM`` -> ``Tumor`` is a defensible merge:
TuPro's ``Normal lymphoid tissue`` is lymphoid tissue, whereas USZ's ``NOR`` is
normal parenchyma and USZ's lymphoid equivalent is ``TLS``, so merging on name
similarity would assert a biological identity that does not hold.
"""

from __future__ import annotations

import numpy as np

#: Written over the annotation of a class that falls under the support floor (see
#: :func:`class_floor_mask`). It is a member of :data:`UNLABELED`, so demoting a
#: class removes it from every label-dependent metric by the same rule that
#: removes an unannotated spot — but it stays distinguishable from "the
#: pathologist wrote nothing here", and ``annotation_raw`` still holds the
#: original name.
BELOW_FLOOR = "BELOW_CLASS_FLOOR"

#: Annotation values that mean "no pathologist label here", across cohorts.
UNLABELED: frozenset[str] = frozenset(
    {"UNASSIGNED", "UNASSIGNED_", "NONE", "NAN", "", "MIXED", "WHITESPACE", "UNKNOWN", BELOW_FLOOR}
)

#: Cohort spelling -> canonical class name, keyed by the upper-cased raw label.
#: Add an entry only when the two names denote the same class beyond doubt.
CANONICAL: dict[str, str] = {
    "TUM": "Tumor",
}


def labeled_mask(labels) -> np.ndarray:
    """Boolean mask selecting spots that carry a real pathology annotation."""
    arr = np.asarray(labels, dtype=object)
    return np.array(
        [(x is not None) and (str(x).strip().upper() not in UNLABELED) for x in arr],
        dtype=bool,
    )


def cohort_classes(table) -> list[str]:
    """The vocabulary every macro average in this pipeline runs over.

    One definition, because it is the *denominator*: ``eval``, ``ablate`` and
    ``integrate`` all average F1 over these classes, and a stage that derived a
    different set would produce numbers that look comparable and are not. The three
    of them each spelled this out for themselves — two of them off ``table.meta``
    and one off ``table.annotation`` — which is three chances for the definition to
    drift and no way to notice if it did.

    The whole annotated cohort, not the classes a level's held-out slides happen to
    carry: a prediction outside the averaged subset still has to count as a false
    negative for its true class. Which subset a given scope averages over is a
    separate question, and :func:`vgtfm.evaluate.reporting.score_vocabulary` answers
    it from the pooled ground truth.
    """
    labels = np.asarray(table.annotation)
    return sorted(set(labels[labeled_mask(labels)].astype(str)))


def canonical(label) -> str:
    """Canonical name for one annotation value.

    Unlabelled values and labels with no :data:`CANONICAL` entry are returned
    unchanged (stripped), so this is safe to apply to a whole column.
    """
    text = "" if label is None else str(label).strip()
    if text.upper() in UNLABELED:
        return text
    return CANONICAL.get(text.upper(), text)


def canonical_labels(labels) -> np.ndarray:
    """Vectorised :func:`canonical` over an annotation column.

    Returns an object array so it can be assigned straight back onto a pandas
    column without widening a fixed-width unicode dtype.
    """
    arr = np.asarray(labels, dtype=object)
    return np.array([canonical(x) for x in arr], dtype=object)


def real_classes(labels) -> list[str]:
    """Sorted unique labelled classes present in *labels*."""
    arr = np.asarray(labels).astype(str)
    return sorted({c for c in arr if c.strip().upper() not in UNLABELED})


def class_floor_mask(
    annotation, tissue, sample_id, *, min_spots: int, min_slides: int
) -> tuple[np.ndarray, list[dict]]:
    """Rows whose class is too thinly supported *within its own tissue* to score.

    A class is kept when it has at least *min_spots* annotated spots and appears on
    at least *min_slides* slides of that tissue. Both counts are per tissue, because
    every probe is fitted within one tissue: a class carried by a single slide there
    cannot be held out and predicted, so its F1 is structurally zero and it enters
    the pooled macro average as a constant that no representation can move.

    The threshold is deliberately blunt and applied before any model is fitted, so
    it cannot be tuned against a result. The two bounds are independent: ``0``
    switches that one off and leaves the other in force, and ``0`` for both
    disables the floor. Returns the boolean mask of rows to demote and one record
    per dropped class.
    """
    annotation = np.asarray(annotation, dtype=object)
    tissue = np.asarray(tissue, dtype=object).astype(str)
    sample_id = np.asarray(sample_id, dtype=object).astype(str)
    drop = np.zeros(len(annotation), dtype=bool)
    if min_spots <= 0 and min_slides <= 0:
        return drop, []

    labeled = labeled_mask(annotation)
    names = annotation.astype(str)
    dropped: list[dict] = []
    for key in sorted({(t, c) for t, c in zip(tissue[labeled], names[labeled])}):
        t, c = key
        rows = labeled & (tissue == t) & (names == c)
        n_spots, n_slides = int(rows.sum()), len(set(sample_id[rows]))
        if n_spots >= min_spots and n_slides >= min_slides:
            continue
        drop |= rows
        dropped.append(
            {
                "tissue": t,
                "annotation": c,
                "n_spots": n_spots,
                "n_slides": n_slides,
                "reason": "spots" if n_spots < min_spots else "slides",
            }
        )
    return drop, dropped
