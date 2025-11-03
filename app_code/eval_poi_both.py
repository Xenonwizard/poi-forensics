# eval_poi_both.py
import argparse, glob, os, numpy as np

def load_globals(glob_pattern: str):
    paths = sorted(glob.glob(glob_pattern))
    scores = []
    for p in paths:
        try:
            d = dict(np.load(p))
            if 'global_score' in d:
                scores.append(float(d['global_score']))
        except Exception as e:
            print(f"[warn] skipping {p}: {e}")
    return paths, scores

def compute_T(scores, perc=10):
    if not scores:
        return None
    return float(np.percentile(scores, perc))

def classify_file(npz_path, T):
    d = dict(np.load(npz_path))
    gs = float(d['global_score'])
    return gs, ("POI" if (T is not None and gs >= T) else "NOT-POI")

def batch_eval(title, val_glob, test_glob_or_file, perc=10):
    print(f"\n=== {title} ===")
    print(f"Validation glob: {val_glob}")
    val_paths, val_scores = load_globals(val_glob)
    print(f"Found {len(val_paths)} validation files with global_score.")
    if not val_scores:
        print("ERROR: No usable validation scores found. Provide .npz outputs from main_test.py.")
        return

    T = compute_T(val_scores, perc=perc)
    print(f"Threshold T (@{perc}th percentile) = {T:.6f}")

    # Accept either a single file or a glob/folder
    if os.path.isdir(test_glob_or_file):
        test_glob = os.path.join(test_glob_or_file, "*.npz")
    elif any(ch in test_glob_or_file for ch in "*?[]"):
        test_glob = test_glob_or_file
    else:
        test_glob = test_glob_or_file  # single file path

    test_paths = sorted(glob.glob(test_glob)) if test_glob != test_glob_or_file or os.path.isdir(test_glob_or_file) else [test_glob_or_file]
    print(f"Testing: {len(test_paths)} file(s)")
    if not test_paths:
        print("WARNING: no test files matched.")
        return

    for p in test_paths:
        try:
            gs, decision = classify_file(p, T)
            print(f"{os.path.basename(p)}  |  global_score={gs:.6f}  ->  {decision}")
        except Exception as e:
            print(f"[warn] could not read {p}: {e}")

def main():
    ap = argparse.ArgumentParser(description="Evaluate POI/not-POI for two apps separately (poiforensics & idreveal).")
    ap.add_argument("--val_poif", required=False, default="/tmp/val_runs_poif/*.npz",
                    help="Glob for validation results (poiforensics).")
    ap.add_argument("--test_poif", required=False, default="/tmp/test_runs_poif/",
                    help="File, folder, or glob for test results (poiforensics).")
    ap.add_argument("--val_idr", required=False, default="/tmp/val_runs_idr/*.npz",
                    help="Glob for validation results (idreveal).")
    ap.add_argument("--test_idr", required=False, default="/tmp/test_runs_idr/",
                    help="File, folder, or glob for test results (idreveal).")
    ap.add_argument("--percentile", type=float, default=10.0,
                    help="Percentile for threshold (e.g., 10 for Pd@10%% false-alarm target).")
    args = ap.parse_args()

    # Evaluate each app independently
    batch_eval("POI-FORENSICS (audio+video)", args.val_poif, args.test_poif, perc=args.percentile)
    batch_eval("ID-REVEAL (video-only)", args.val_idr, args.test_idr, perc=args.percentile)

if __name__ == "__main__":
    main()
