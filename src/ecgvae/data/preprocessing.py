"""
Stage 1: Filter MIT-BIH Arrhythmia Database records to those with MLII on channel 0.
Stage 2: Extract fixed-length beat windows centered on annotated R-peaks for
         those records, with raw MIT-BIH symbol + AAMI EC57 class label.

Outputs:
  data/processed/mlii_channel0_records.json   (stage 1)
  data/processed/beat_windows.npy             (stage 2, shape (N, window_len), float32)
  data/processed/beat_metadata.csv            (stage 2, one row per window)
"""

import wfdb
import os
import json
import csv
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "mit-bih-arrhythmia-database-1.0.0"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# ---------------------------------------------------------------------------
# Stage 1: record filtering
# ---------------------------------------------------------------------------

# Known exception in MIT-BIH: records 201 and 202 are the SAME patient
# (same subject, recorded at two different times). Every other record
# is a distinct patient. We build patient_id explicitly so downstream
# patient-wise splitting doesn't leak this pair across train/test.
DUPLICATE_PATIENT_PAIRS = {
    "202": "201",  # map 202 -> same patient_id as 201
}

# Records containing paced beats. AAMI-recommended practice is to exclude
# these from evaluation, since pacemaker artifacts have distinct signal
# characteristics not representative of natural cardiac activity.
# Source: PhysioNet MIT-BIH directory + AAMI EC57 recommended practice,
# confirmed against multiple published evaluation protocols.
PACED_RECORDS = {"102", "104", "107", "217"}


def get_all_record_names(pn_dir="mitdb", local_dir=None):
    """Get the list of all record names in the database."""
    if local_dir is not None:
        names = sorted({
            f.split(".")[0]
            for f in os.listdir(local_dir)
            if f.endswith(".hea")
        })
        return names
    else:
        return wfdb.get_record_list(pn_dir)


def build_patient_id(record_name):
    """Map record_name -> patient_id, merging known duplicate-patient records."""
    return DUPLICATE_PATIENT_PAIRS.get(record_name, record_name)


def filter_mlii_channel0(pn_dir="mitdb", local_dir=None, verbose=True,
                          exclude_paced_records=True):
    """
    Return list of records where channel 0 (signal index 0) is MLII.

    Args:
        exclude_paced_records: if True (default), also drop the 4 records
            known to contain paced beats (102, 104, 107, 217), following
            AAMI-recommended evaluation practice. Set False if you have a
            specific reason to keep them (e.g. paced beats as a deliberate
            disentanglement test case) -- if so, document that choice.
    """
    record_names = get_all_record_names(pn_dir=pn_dir, local_dir=local_dir)
    results = []
    excluded = []

    for rec in record_names:
        if exclude_paced_records and rec in PACED_RECORDS:
            excluded.append((rec, "contains paced beats (AAMI exclusion)"))
            continue

        try:
            if local_dir is not None:
                header = wfdb.rdheader(os.path.join(local_dir, rec))
            else:
                header = wfdb.rdheader(rec, pn_dir=pn_dir)
        except Exception as e:
            if verbose:
                print(f"[WARN] could not read header for {rec}: {e}")
            continue

        sig_names = header.sig_name

        if len(sig_names) == 0:
            excluded.append((rec, "no channels found"))
            continue

        if sig_names[0] == "MLII":
            results.append({
                "record_id": rec,
                "patient_id": build_patient_id(rec),
                "sig_name": sig_names,
                "fs": header.fs,
                "n_sig": header.n_sig,
            })
        else:
            excluded.append((rec, f"channel 0 = {sig_names[0]}"))

    if verbose:
        print(f"Total records found:      {len(record_names)}")
        print(f"Records with MLII as ch0: {len(results)}")
        print(f"Excluded records:         {len(excluded)}")
        for rec, reason in excluded:
            print(f"  - {rec}: {reason}")
        n_patients = len(set(r["patient_id"] for r in results))
        print(f"Unique patients in filtered set: {n_patients}")

    return results


# ---------------------------------------------------------------------------
# Stage 2: beat window extraction
# ---------------------------------------------------------------------------

# Window size around R-peak, in samples, at the native 360 Hz sampling rate.
# Asymmetric: more room after R to capture the T-wave.
WINDOW_PRE = 100   # ~0.278 s before R-peak
WINDOW_POST = 144  # ~0.400 s after R-peak
WINDOW_LEN = WINDOW_PRE + WINDOW_POST  # 244 samples

# AAMI EC57 superclass mapping. Symbols not in this dict are treated as
# non-beat annotations (rhythm/quality markers etc.) and skipped entirely.
AAMI_MAP = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
    "F": "F",
    "/": "Q", "f": "Q", "Q": "Q",
}


def extract_beat_windows(record_id, patient_id, raw_dir,
                          w_pre=WINDOW_PRE, w_post=WINDOW_POST, verbose=True):
    """
    Extract fixed-length windows around every annotated beat in one record.

    Returns:
        windows: list of np.ndarray, each shape (w_pre + w_post,)
        meta_rows: list of dicts, one per window, aligned with `windows`
        n_dropped_boundary: int, beats skipped because window ran off the edge
        n_skipped_nonbeat: int, annotations skipped because not a beat symbol
    """
    rec_path = str(Path(raw_dir) / record_id)
    record = wfdb.rdrecord(rec_path)
    ann = wfdb.rdann(rec_path, "atr")

    signal = record.p_signal[:, 0]  # channel 0, MLII (guaranteed by stage 1 filter)
    sig_len = len(signal)

    windows = []
    meta_rows = []
    n_dropped_boundary = 0
    n_skipped_nonbeat = 0

    for sample_idx, symbol in zip(ann.sample, ann.symbol):
        aami_label = AAMI_MAP.get(symbol, None)
        if aami_label is None:
            n_skipped_nonbeat += 1
            continue

        start = sample_idx - w_pre
        end = sample_idx + w_post
        if start < 0 or end > sig_len:
            n_dropped_boundary += 1
            continue

        window = signal[start:end].astype(np.float32)
        windows.append(window)
        meta_rows.append({
            # window_idx is filled in later, once windows from all records
            # are concatenated -- do not rely on positional order alone,
            # join on this column explicitly downstream.
            "window_idx": None,
            "record_id": record_id,
            "patient_id": patient_id,
            "r_peak_sample": int(sample_idx),
            "raw_symbol": symbol,
            "aami_label": aami_label,
        })

    if verbose:
        print(f"  {record_id}: {len(windows)} beats kept, "
              f"{n_dropped_boundary} dropped (boundary), "
              f"{n_skipped_nonbeat} skipped (non-beat annotation)")

    return windows, meta_rows, n_dropped_boundary, n_skipped_nonbeat


def build_beat_dataset(filtered_records, raw_dir, verbose=True):
    """Run extraction across all filtered records and concatenate results."""
    all_windows = []
    all_meta = []
    total_dropped_boundary = 0
    total_skipped_nonbeat = 0

    for rec in filtered_records:
        windows, meta_rows, n_drop, n_skip = extract_beat_windows(
            rec["record_id"], rec["patient_id"], raw_dir, verbose=verbose
        )
        all_windows.extend(windows)
        all_meta.extend(meta_rows)
        total_dropped_boundary += n_drop
        total_skipped_nonbeat += n_skip

    windows_array = np.stack(all_windows, axis=0) if all_windows else np.empty((0, WINDOW_LEN))

    # Assign explicit window_idx now that final order is fixed. Downstream
    # code should join windows_array[i] to metadata via this column, not
    # via bare positional trust.
    for i, row in enumerate(all_meta):
        row["window_idx"] = i

    if verbose:
        print(f"\nTotal beats extracted: {len(all_windows)}")
        print(f"Total dropped (boundary): {total_dropped_boundary}")
        print(f"Total skipped (non-beat annotation): {total_skipped_nonbeat}")
        print(f"Windows array shape: {windows_array.shape}")

    return windows_array, all_meta


def print_class_summary(meta_rows):
    """Print AAMI class distribution and per-record beat counts."""
    aami_counts = {}
    raw_counts = {}
    for row in meta_rows:
        aami_counts[row["aami_label"]] = aami_counts.get(row["aami_label"], 0) + 1
        raw_counts[row["raw_symbol"]] = raw_counts.get(row["raw_symbol"], 0) + 1

    total = len(meta_rows)
    print("\n--- AAMI class distribution ---")
    for cls in sorted(aami_counts, key=lambda c: -aami_counts[c]):
        n = aami_counts[cls]
        print(f"  {cls}: {n:6d}  ({100 * n / total:5.2f}%)")

    print("\n--- Raw MIT-BIH symbol distribution ---")
    for sym in sorted(raw_counts, key=lambda s: -raw_counts[s]):
        n = raw_counts[sym]
        print(f"  {sym}: {n:6d}  ({100 * n / total:5.2f}%)")

    n_patients = len(set(row["patient_id"] for row in meta_rows))
    n_records = len(set(row["record_id"] for row in meta_rows))
    print(f"\nTotal beats: {total}")
    print(f"Records represented: {n_records}")
    print(f"Patients represented: {n_patients}")


def save_beat_dataset(windows_array, meta_rows, processed_dir):
    processed_dir.mkdir(parents=True, exist_ok=True)

    npy_path = processed_dir / "beat_windows.npy"
    np.save(npy_path, windows_array)

    csv_path = processed_dir / "beat_metadata.csv"
    fieldnames = ["window_idx", "record_id", "patient_id", "r_peak_sample", "raw_symbol", "aami_label"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(meta_rows)

    print(f"\nSaved windows to {npy_path}")
    print(f"Saved metadata to {csv_path}")
    print("Row i in beat_metadata.csv corresponds to windows_array[i].")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Set to False if you have a deliberate reason to keep paced-beat
    # records (102, 104, 107, 217) -- default follows AAMI recommended
    # practice and excludes them.
    EXCLUDE_PACED_RECORDS = True

    # Stage 1
    filtered_records = filter_mlii_channel0(
        local_dir=str(RAW_DIR), exclude_paced_records=EXCLUDE_PACED_RECORDS
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    records_out_path = PROCESSED_DIR / "mlii_channel0_records.json"
    with open(records_out_path, "w") as f:
        json.dump(filtered_records, f, indent=2)
    print(f"\nWrote {len(filtered_records)} records to {records_out_path}\n")

    # Stage 2
    print("Extracting beat windows...")
    windows_array, meta_rows = build_beat_dataset(filtered_records, RAW_DIR)
    save_beat_dataset(windows_array, meta_rows, PROCESSED_DIR)
    print_class_summary(meta_rows)