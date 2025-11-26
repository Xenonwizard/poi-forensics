import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import erf


def gaus2prob(x):
    return (1 - erf(x / np.sqrt(2.0))) / 2


def compute_statistics(all_ranges, all_dists, global_score, dist_normalization):
    """
    Computes:
        1. percentage of frames with face detected
        2. percentage of values below the global score
    """

    # ---------- FACE DETECTION PERCENTAGE ----------
    total_frames = len(all_dists)
    face_detected_frames = np.sum(~np.isnan(all_dists))
    pct_face_detected = 100.0 * face_detected_frames / total_frames

    # ---------- BELOW GLOBAL SCORE PERCENTAGE ----------
    if global_score is not None:
        valid_dists = all_dists[~np.isnan(all_dists)]

        if dist_normalization:
            global_score_val = gaus2prob(global_score)
            dists_val = gaus2prob(valid_dists)
        else:
            global_score_val = global_score
            dists_val = valid_dists

        pct_below_global = 100.0 * np.sum(dists_val < global_score_val) / len(dists_val)
    else:
        pct_below_global = None

    return pct_face_detected, pct_below_global


def create_plot(dict_out, output_image, dist_normalization):

    embs_track = np.asarray(dict_out['embs_track'])
    embs_dists = np.asarray(dict_out['embs_dists'])
    embs_range = np.asarray(dict_out['embs_range'])

    # flatten dist if needed
    while len(embs_dists.shape) > 1:
        embs_dists = embs_dists[..., -1]

    # detect global score
    global_score = dict_out.get('global_score', None)

    # ---- store all values for statistics later ----
    all_ranges = []
    all_dists = []

    xmin = np.PINF
    xmax = 0
    fig = plt.figure(figsize=(12, 6))

    for ids in np.unique(embs_track):
        inds = embs_track == ids
        dist = embs_dists[inds]
        rang = np.mean(embs_range[inds], -1) / 25.0

        # accumulate for stats
        all_ranges.append(rang)
        all_dists.append(dist)

        xmin = min(xmin, np.min(rang))
        xmax = max(xmax, np.max(rang))

        if dist_normalization:
            plt.semilogy(rang, gaus2prob(dist), 'k', linewidth=2)
        else:
            plt.plot(rang, dist, 'k', linewidth=2)

    # merge arrays for statistics
    all_ranges = np.concatenate(all_ranges)
    all_dists = np.concatenate(all_dists)

    # ---------- draw global score line ----------
    if global_score is not None:
        if dist_normalization:
            plt.hlines(
                gaus2prob(global_score), xmin, xmax,
                'b', linestyles='dashdot', label='global_score'
            )
        else:
            plt.hlines(
                global_score, xmin, xmax,
                'b', linestyles='dashdot', label='global_score'
            )

    plt.grid()
    plt.legend()
    plt.xlabel('time')

    if dist_normalization:
        plt.gca().invert_yaxis()
    else:
        plt.ylabel('distance')

    fig.savefig(output_image)

    # ---------- compute and PRINT statistics ----------
    pct_face_detected, pct_below = compute_statistics(
        all_ranges, all_dists, global_score, dist_normalization
    )

    print("\n========== FINAL STATISTICS ==========")
    print(f"Face detected percentage: {pct_face_detected:.2f}%")
    if pct_below is not None:
        print(f"Percentage of time distance < global score: {pct_below:.2f}%")
    else:
        print("Global score not available — skipping threshold percentage.")
    print("======================================\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='create plot.'
    )
    parser.add_argument('--file_npz', type=str, help='numpy file.')
    parser.add_argument('--output_image', type=str, help='output image (.png).')
    parser.add_argument('--dist_normalization', type=int, default=0,
                        help="if True, normalize distances using pristine video values.")
    argd = parser.parse_args()

    create_plot(np.load(argd.file_npz), argd.output_image, argd.dist_normalization)
