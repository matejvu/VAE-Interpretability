"""
Assign train / val / test splits to the extracted beat dataset.

Split logic:
  - DS2 (de Chazal et al. 2004 inter-patient split) is held out entirely as
    TEST. It is never touched during VAE training or hyperparameter
    selection -- not even in an unsupervised capacity -- so that any later
    downstream evaluation (classification, interpretability generalization)
    is a genuine unseen-patient test, comparable to the wider literature.
  - DS1 is the train pool. It is further split into TRAIN / VAL by patient
    using StratifiedGroupKFold, so validation beats come from held-out
    patients (no leakage) while still trying to preserve the AAMI class
    balance across the split (important given severe class imbalance and
    a small number of patients).

Note: the literature DS1 list includes record 114. Our filtered dataset
does not contain 114 (its channel 0 is V5, not MLII -- excluded at the
MLII-filtering stage). This is expected: 114 simply will not appear in
the output, no special-casing needed.

Output:
  data/processed/patient_splits.json   -- {patient_id: "train"|"val"|"test"}
"""

import json
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# de Chazal et al. (2004) inter-patient split, as used across the
# MIT-BIH AAMI-classification literature for comparability.
DS1_RECORDS = {
    "101", "106", "108", "109", "112", "114", "115", "116", "118", "119",
    "122", "124", "201", "203", "205", "207", "208", "209", "215", "220",
    "223", "230",
}
DS2_RECORDS = {
    "100", "103", "105", "111", "113", "117", "121", "123", "200", "202",
    "210", "212", "213", "214", "219", "221", "222", "228", "231", "232",
    "233", "234",
}

# Fraction of DS1 patients held out as validation (approx; StratifiedGroupKFold
# operates in folds, so this is achieved via fold count, not a direct fraction).
N_VAL_FOLDS = 5  # 1/5 of DS1 patients -> val, rest -> train
RANDOM_STATE = 42


def assign_ds1_ds2(record_id):
    if record_id in DS1_RECORDS:
        return "DS1"
    elif record_id in DS2_RECORDS:
        return "DS2"
    else:
        raise ValueError(
            f"record_id {record_id} not found in either DS1 or DS2 -- "
            f"unexpected record, check paced-record / channel exclusions."
        )


def build_split(meta_csv_path, random_state=RANDOM_STATE, n_val_folds=N_VAL_FOLDS):
    df = pd.read_csv(meta_csv_path, dtype={"record_id": str, "patient_id": str})

    df["ds"] = df["record_id"].apply(assign_ds1_ds2)

    # Known conflict: the classic de Chazal DS1/DS2 split predates
    # patient-aware handling of records 201/202, which are the SAME
    # patient but were assigned to different sets (201 -> DS1, 202 -> DS2).
    # Since we merge them into one patient_id for leakage prevention, this
    # patient would otherwise straddle train/val pool AND test -- a real
    # leak. Resolution: force all of that patient's records into DS1,
    # since patient-level leakage prevention takes priority over exact
    # literature record-list parity. This is a single, fully documented
    # deviation affecting one patient out of 42.
    conflict_patients = (
        df.groupby("patient_id")["ds"].nunique().loc[lambda s: s > 1].index.tolist()
    )
    if conflict_patients:
        for pid in conflict_patients:
            affected_records = df.loc[df["patient_id"] == pid, "record_id"].unique().tolist()
            print(f"[SPLIT CONFLICT] patient_id={pid} spans both DS1 and DS2 "
                  f"(records: {affected_records}). Forcing all of this "
                  f"patient's records into DS1 to prevent train/test leakage.")
        df.loc[df["patient_id"].isin(conflict_patients), "ds"] = "DS1"

    # Sanity check: every record actually present in our data should map
    # cleanly into DS1 or DS2 (paced records and 114 are simply absent,
    # not mismatched).
    n_ds1 = df.loc[df["ds"] == "DS1", "record_id"].nunique()
    n_ds2 = df.loc[df["ds"] == "DS2", "record_id"].nunique()
    print(f"Records present from DS1: {n_ds1} (of 22 in literature list)")
    print(f"Records present from DS2: {n_ds2} (of 22 in literature list)")

    ds1_df = df[df["ds"] == "DS1"].copy()

    # Diagnostic: per-patient class counts within DS1. Rare-class beats
    # (esp. F) tend to concentrate in a handful of patients -- if so, no
    # automatic fold-selection can guarantee good F representation in both
    # train and val simultaneously (there just aren't enough F-carrying
    # patients to distribute). Better to see this directly than discover
    # it only through fold outcomes.
    print("\n--- Per-patient AAMI class counts within DS1 (diagnostic) ---")
    patient_class_counts = ds1_df.groupby(["patient_id", "aami_label"]).size().unstack(fill_value=0)
    patient_class_counts = patient_class_counts.reindex(
        columns=sorted(patient_class_counts.columns), fill_value=0
    )
    print(patient_class_counts.sort_values("F", ascending=False))
    print()

    # StratifiedGroupKFold: groups = patient_id (keeps a patient entirely in
    # one fold), stratify on aami_label (tries to preserve class balance
    # across folds despite the small number of groups).
    sgkf = StratifiedGroupKFold(n_splits=n_val_folds, shuffle=True, random_state=random_state)

    X_placeholder = ds1_df.index.values
    y = ds1_df["aami_label"].values
    groups = ds1_df["patient_id"].values

    # Rather than blindly taking the first fold, generate all candidate
    # folds and pick the one whose validation class distribution is closest
    # to the overall DS1 class distribution. With only 21 DS1 patient-groups,
    # stratification is coarse-grained -- rare classes (esp. F) can cluster
    # in specific patients, so which fold you land on matters a lot for
    # whether validation is actually usable for that class.
    #
    # NOTE: plain L1/L2 distance on raw proportions is the WRONG metric here
    # -- a rare class (F, ~0.8% overall) can only ever deviate by a small
    # absolute amount, so it's structurally outweighed by ordinary sampling
    # noise in the majority class (N, ~89%). L2 makes this worse, not
    # better, since squaring further shrinks the already-small rare-class
    # terms relative to majority-class terms. Instead we use a chi-squared-
    # style distance, dividing each squared deviation by that class's own
    # expected (overall) proportion -- this upweights deviations in rare
    # classes exactly where L1/L2 would ignore them.
    overall_dist = ds1_df["aami_label"].value_counts(normalize=True)

    best_fold = None
    best_score = None
    fold_candidates = list(sgkf.split(X_placeholder, y, groups))

    for fold_i, (train_idx, val_idx) in enumerate(fold_candidates):
        val_labels = ds1_df.iloc[val_idx]["aami_label"]
        val_dist = val_labels.value_counts(normalize=True).reindex(overall_dist.index, fill_value=0.0)
        chi2_dist = (((val_dist - overall_dist) ** 2) / overall_dist).sum()
        val_counts = val_labels.value_counts().reindex(overall_dist.index, fill_value=0)
        print(f"  fold {fold_i}: val beats={len(val_idx)}, chi2 dist={chi2_dist:.4f}, "
              f"class counts={val_counts.to_dict()}")
        if best_score is None or chi2_dist < best_score:
            best_score = chi2_dist
            best_fold = fold_i

    print(f"Selected fold {best_fold} (lowest chi-squared distance to overall DS1 class distribution)")
    train_idx, val_idx = fold_candidates[best_fold]
    val_patient_ids = set(ds1_df.iloc[val_idx]["patient_id"])
    train_patient_ids = set(ds1_df.iloc[train_idx]["patient_id"])

    # Sanity check: no patient should appear in both (StratifiedGroupKFold
    # guarantees this, but verify explicitly rather than trust blindly).
    overlap = train_patient_ids & val_patient_ids
    assert not overlap, f"Patient leakage between train/val: {overlap}"

    test_patient_ids = set(df[df["ds"] == "DS2"]["patient_id"])

    # Sanity check: train/val/test patient sets are pairwise disjoint.
    assert not (train_patient_ids & test_patient_ids)
    assert not (val_patient_ids & test_patient_ids)

    patient_splits = {}
    for pid in train_patient_ids:
        patient_splits[pid] = "train"
    for pid in val_patient_ids:
        patient_splits[pid] = "val"
    for pid in test_patient_ids:
        patient_splits[pid] = "test"

    return patient_splits, df


def print_split_summary(patient_splits, df):
    df = df.copy()
    df["split"] = df["patient_id"].map(patient_splits)

    print("\n--- Patients per split ---")
    for split in ["train", "val", "test"]:
        n_patients = sum(1 for s in patient_splits.values() if s == split)
        print(f"  {split}: {n_patients} patients")

    print("\n--- Beats per split ---")
    print(df["split"].value_counts())

    print("\n--- AAMI class distribution per split (%) ---")
    ct = pd.crosstab(df["split"], df["aami_label"], normalize="index") * 100
    print(ct.round(2))

    print("\n--- Raw beat counts per split / class ---")
    ct_counts = pd.crosstab(df["split"], df["aami_label"])
    print(ct_counts)


if __name__ == "__main__":
    meta_csv_path = PROCESSED_DIR / "beat_metadata.csv"
    patient_splits, df = build_split(meta_csv_path)

    out_path = PROCESSED_DIR / "patient_splits.json"
    with open(out_path, "w") as f:
        json.dump(patient_splits, f, indent=2, sort_keys=True)
    print(f"\nWrote patient split mapping to {out_path}")

    print_split_summary(patient_splits, df)