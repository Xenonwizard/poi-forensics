import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest, norm


# ======================================================
# USER CONFIGURATION — EDIT THESE
# ======================================================

NPZ_INPUT = "./norm_dist.npz"              # path to your NPZ file
PLOT_OUTPUT = "./similarity_distribution.png"  # output plot file

FALSE_ALARM_RATE = 0.10   # Pfa = 10% (paper default)

# ======================================================
# INTERNAL FUNCTIONS
# ======================================================

def load_scores(npz_path):
    """
    Extract per-segment similarity scores S_m^{POI}(c)
    from the 'embs_dists' array stored in NPZ.
    """
    data = np.load(npz_path, allow_pickle=True)

    if "embs_dists" not in data:
        raise KeyError(f"'embs_dists' not found in NPZ file: {npz_path}")

    scores = np.asarray(data["embs_dists"]).squeeze()

    # If extra dimensions exist, flatten to 1D
    while scores.ndim > 1:
        scores = scores[..., -1]

    return scores


def gaussian_normalize(scores):
    """Normalize scores into zero-mean, unit-variance (z-scoring)."""
    mu = np.mean(scores)
    sigma = np.std(scores)
    normalized = (scores - mu) / sigma
    return normalized, mu, sigma


def test_gaussian(normalized_scores):
    """
    Kolmogorov-Smirnov test to compare distribution against N(0,1).
    """
    ks_stat, p_value = kstest(normalized_scores, norm.cdf)
    return ks_stat, p_value


def compute_classification(normalized_scores, pfa=0.10):
    """
    Compute threshold and classify the video as REAL or FAKE.
    """
    threshold = norm.ppf(pfa)      # for pfa=10%, threshold ≈ -1.2816
    S_avg = np.mean(normalized_scores)

    verdict = "REAL" if S_avg >= threshold else "FAKE"

    return S_avg, threshold, verdict


def plot_distribution(normalized_scores, filename):
    """Create histogram + overlay theoretical Gaussian curve."""
    plt.figure(figsize=(8, 5))

    plt.hist(normalized_scores, bins=10, density=True, alpha=0.6, label="Similarity Scores")

    x = np.linspace(-4, 4, 200)
    plt.plot(x, norm.pdf(x), "r--", label="N(0,1)")

    plt.title("Distribution of Normalized POI Similarity Scores")
    plt.xlabel("Normalized Score")
    plt.ylabel("Density")
    plt.legend()
    plt.grid()

    plt.savefig(filename)
    print(f"[+] Plot saved to {filename}")


# ======================================================
# MAIN EXECUTION
# ======================================================
if __name__ == "__main__":
    print("\n=== POI Gaussian Evaluation (Paper Method) ===")

    # 1 — Load similarity scores
    scores = load_scores(NPZ_INPUT)
    print(f"[+] Loaded {len(scores)} segment scores")

    # 2 — Normalize scores to test Gaussian behavior
    normalized, mu, sigma = gaussian_normalize(scores)
    print(f"[+] Raw Mean = {mu:.4f}, Raw Std = {sigma:.4f}")
    print(f"[+] Normalized Mean = {np.mean(normalized):.4f}, Std = {np.std(normalized):.4f}")

    # 3 — Gaussian normality test
    ks_stat, p_value = test_gaussian(normalized)
    print(f"[+] KS Test Statistic = {ks_stat:.4f}, p-value = {p_value:.4f}")

    if p_value > 0.05:
        print("    ✔ Distribution behaves like Gaussian → Consistent with REAL POI video")
    else:
        print("    ✘ Distribution NOT Gaussian → Potentially FAKE video")

    # 4 — Decision threshold according to Pfa = 10%
    avg_score, threshold, verdict = compute_classification(normalized, FALSE_ALARM_RATE)
    print(f"\n[+] Average Score = {avg_score:.4f}")
    print(f"[+] Threshold (Pfa={FALSE_ALARM_RATE*100:.0f}%) = {threshold:.4f}")
    print(f"\n🔥 FINAL CLASSIFICATION: {verdict}")

    # 5 — Plot distribution
    plot_distribution(normalized, PLOT_OUTPUT)
