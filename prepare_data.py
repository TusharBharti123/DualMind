#!/usr/bin/env python
"""Verify preprocessed SWaT/SMD arrays and write a compact dataset report."""
from pathlib import Path
import argparse
import csv
import numpy as np

def load_npz(path):
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}

def inspect_file(path, kind, dataset, machine=""):
    data = load_npz(path)
    X = data.get("X")
    if X is None:
        raise ValueError(f"{path}: expected key 'X'; found {list(data)}")
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"{path}: X should be 2-D, got {X.shape}")
    finite = np.isfinite(X)
    row = {
        "dataset": dataset,
        "machine": machine,
        "split": kind,
        "path": str(path),
        "rows": int(X.shape[0]),
        "features": int(X.shape[1]),
        "nan_values": int(np.isnan(X).sum()),
        "inf_values": int(np.isinf(X).sum()),
        "nonfinite_rows": int((~finite.all(axis=1)).sum()),
        "constant_features": int(np.sum(np.nanstd(X, axis=0) < 1e-12)),
    }
    if "y" in data:
        y = np.asarray(data["y"]).reshape(-1)
        if len(y) != len(X):
            raise ValueError(f"{path}: label length {len(y)} != rows {len(X)}")
        vals, counts = np.unique(y, return_counts=True)
        row["label_counts"] = ";".join(f"{v}:{int(c)}" for v, c in zip(vals, counts))
        row["anomaly_rows"] = int(np.sum(y.astype(int) == 1))
    else:
        row["label_counts"] = ""
        row["anomaly_rows"] = ""
    if "timestamps" in data:
        row["timestamp_count"] = int(len(data["timestamps"]))
        row["timestamp_monotonic"] = bool(
            len(data["timestamps"]) < 2 or np.all(data["timestamps"][1:] >= data["timestamps"][:-1])
        )
    else:
        row["timestamp_count"] = ""
        row["timestamp_monotonic"] = ""
    row["status"] = "OK" if row["nonfinite_rows"] == 0 else "CHECK_NONFINITE"
    return row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="processed", help="Preprocessing output directory")
    ap.add_argument("--out", default="prepared", help="Report output directory")
    args = ap.parse_args()
    root, out = Path(args.input), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    swat = root / "swat"
    if swat.exists():
        for name, kind in [("train_normal.npz", "train_normal"), ("test.npz", "test")]:
            p = swat / name
            if p.exists():
                rows.append(inspect_file(p, kind, "SWaT"))
    smd = root / "smd"
    if smd.exists():
        for machine_dir in sorted(p for p in smd.glob("machine-*") if p.is_dir()):
            for name, kind in [("train_normal.npz", "train_normal"), ("test.npz", "test")]:
                p = machine_dir / name
                if p.exists():
                    rows.append(inspect_file(p, kind, "SMD", machine_dir.name))
    if not rows:
        raise FileNotFoundError(f"No expected NPZ files found under {root.resolve()}")
    report = out / "data_quality_report.csv"
    fields = sorted({k for r in rows for k in r.keys()})
    with report.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Inspected {len(rows)} files.")
    print(f"Report: {report.resolve()}")
    bad = [r for r in rows if r["status"] != "OK"]
    if bad:
        print("\nWARNING: non-finite feature values found:")
        for r in bad:
            print(f"  {r['dataset']} {r['machine']} {r['split']}: {r['nonfinite_rows']} rows")
    else:
        print("All inspected feature arrays contain only finite values.")
    print("\nSummary:")
    for r in rows:
        print(f"{r['dataset']:4} {r['machine'] or '-':12} {r['split']:12} "
              f"X=({r['rows']:,}, {r['features']}) anomalies={r['anomaly_rows']} "
              f"nonfinite_rows={r['nonfinite_rows']}")

if __name__ == "__main__":
    main()
