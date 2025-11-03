import glob, numpy as np

# 1️⃣ Collect validation scores from real POI videos
# (change these to match where you saved the .npz files for real Nicolas Cage clips)
val_npzs = glob.glob("./output/result1.npz")   # genuine POI validation results
real_scores = []
for f in val_npzs:
    d = dict(np.load(f))
    if 'global_score' in d:
        real_scores.append(float(d['global_score']))

if not real_scores:
    raise ValueError("❌ No validation .npz files found in ./output/val_idreveal/. "
                     "Run main_test.py on a few real POI videos first!")

# 2️⃣ Compute threshold (10th percentile of real scores)
T = np.percentile(real_scores, 10)
print(f"✅ Computed threshold T = {T:.3f} from {len(real_scores)} validation files")

# 3️⃣ Evaluate a new test video
test_path = "./output/result1.npz"    # <-- this is your test .npz
d_test = dict(np.load(test_path))
gs = float(d_test['global_score'])
decision = "POI" if gs >= T else "NOT-POI"
print(f"📄 Test file: {test_path}")
print(f"   Global score = {gs:.3f}")
print(f"👉 Decision: {decision}")
