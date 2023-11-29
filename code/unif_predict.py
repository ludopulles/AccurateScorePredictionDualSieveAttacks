"""
Try to explain the distribution of scores, when targets are drawn uniformly in the torus of space
quotiented by the lattice.
"""
from csv import DictReader
from functools import partial, cache
from mpmath import mp, mpc, ceil, exp, erfc, log, pi, sqrt
from multiprocessing import Pool
from os import cpu_count, path
import sys

import matplotlib.pyplot as plt
import matplotlib.colors as cls
from scipy.integrate import quad
import numpy as np

from utils import average_score_in_ball, ball_distribution, GH


def log2(x):
    return float(log(x)/log(2))

def normal_SF(value, var):
    """
    Return Pr[X >= value], when X is normally distributed with mean 0 and variance var.
    :param value: evaluation point
    :param var: variance of the normal distribution
    :returns: the probability of having X >= value ("survival function").
    """
    return 0.5 * erfc(value / sqrt(2 * var))


def integrand_unif(n, N, score, sat_radius, gh_factor):
    """
    Compute value of the integrand for a given value `gh_factor`, i.e. give the probability a score
    is above `score` when a (primal) target is sampled on a sphere of radius `gh_factor GH(n)`.
    :param n: dimension of primal target and the `N` dual vectors
    :param N: the number of dual vectors used in computing the scores
    :param score: score to compare against
    :param gh_factor: gaussian heuristic factor indicating the radius of the sphere relative to the
    gaussian heuristic.
    :param sat_radius: maximum allowed length of dual vectors in output database after sieving.
    """
    len_target, len_duals = gh_factor * GH(n), sat_radius * GH(n)
    r12 = len_duals * len_target

    avg, var = ball_distribution(n, r12, N)
    return normal_SF(score - avg, var) * n * gh_factor**(n-1)


def predict_unif(n, N, sat_radius, score):
    # Inaccuracy is in quad()[1]
    return quad(partial(integrand_unif, n, N, score, sat_radius), 0.0, 1.0, epsrel=1e-20)[0]


def plot_unif_prediction(pool, n, N, real_scores, real_log2_sf, png_file,
                         sat_radius, write_csv, csv_comments):
    LINE_WIDTH = 0.75
    xmin, xmax = 0.0, float(min(N, 15 * sqrt(N / 2)))
    plt.xlim(xmin, xmax)
    scores = np.linspace(xmin, xmax, 200)

    # Experimental data
    if real_scores is not None:
        plt.plot(real_scores, real_log2_sf, color='tab:red', label='experiment', lw=LINE_WIDTH)

    # Heuristic prediction
    hpreds = [log2(normal_SF(x, 0.5 * N)) for x in scores]
    plt.plot(scores, hpreds, color='black', linestyle='dotted', label='heuristic pred.', lw=LINE_WIDTH)

    # Accurate prediction
    preds = list(map(log2, pool.map(partial(predict_unif, n, N, sat_radius), scores)))
    plt.plot(scores, preds, color='tab:orange', label='prediction', lw=LINE_WIDTH)

    plt.legend(loc='lower left')
    plt.title(f'Experimental data vs predictions (n={n}, N = {N})')
    plt.savefig(png_file, dpi=500)
    plt.close()

    # Write also to a csv:
    if write_csv:
        csv_file, ext = path.splitext(png_file)
        assert ext == '.png'
        csv_file += '.csv'
        csv_comments = ('# ' + "\n# ".join(csv_comments) + "\n") if csv_comments else ''
        if real_scores is None:
            with open(csv_file, 'w', encoding='ascii') as f:
                f.write(f"score,lg_sf_pred,lg_sf_heur\n{csv_comments}")
                for i, score in enumerate(scores):
                    f.write(f"{score:.3f},{preds[i]:.3f},{hpreds[i]:.3f}\n")
        else:
            hpreds = [log2(normal_SF(x, 0.5 * N)) for x in real_scores]
            preds = list(map(log2, pool.map(partial(predict_unif, n, N, sat_radius), real_scores)))
            with open(csv_file, 'w', encoding='ascii') as f:
                f.write(f"score,lg_sf_pred,lg_sf_heur,lg_sf_real\n{csv_comments}")
                for i, score in enumerate(real_scores):
                    f.write(f"{score:.3f},{preds[i]:.3f},{hpreds[i]:.3f},{real_log2_sf[i]:.3f}\n")


def read_experimental_data(file):
    scores, log2_sf = [], []
    metadata = {}
    with open(file, newline='') as csv:
        reader = DictReader(csv)
        for line in reader:
            if line['score'] == '':
                metadata[line['metakey']] = line['metaval']
            elif line['score'][0] == '#':
                key, val = line['score'][2:].split(" = ")
                metadata[key] = val
            else:
                scores.append(float(line['score']))
                log2_sf.append(float(line['lg_sf_real']))
    return scores, log2_sf, metadata


if __name__ == '__main__':
    # Command Line interface has two options:
    # - For given `n`, give predictions in a plot
    # - For a given filename, provide also prediction, assuming experimental data was available.
    if len(sys.argv) == 1:
        print(f"Usage: {sys.argv[0]} [n || filename]...")
        print(f"\tn\n\t\tMakes a prediction in dimension `n`, assuming default sieve.")
        print(f"\tfilename\n\t\tAdds predictions on top of experimental data")
        exit(0)

    mp.prec = 256

    with Pool(cpu_count() // 2) as pool:
        for file in sys.argv[1:]:
            try:
                n = int(file)
            except ValueError:
                n = None # file is actually a file

            if n is not None and isinstance(n, int):
                # Give a prediction for a given dimension `n`.
                SATURATION_RATIO = 0.5
                sat_radius = sqrt(4.0 / 3)
                N = ceil(0.5 * SATURATION_RATIO * sat_radius**n)

                png_file = f"../data/pred_unif_n{n:03}.png"
                csv_comments = [f"n = {n}", f"N = {N}", f"saturation radius = {float(sat_radius):.6f}"]

                plot_unif_prediction(pool, n, N, None, None, png_file, sat_radius, True, csv_comments)
            else:
                # Add predictions to experimental data.
                scores, log2_sf, metadata = read_experimental_data(file)
                n = int(metadata['n'])
                N = int(metadata['N'])
                sat_radius = float(metadata.get('saturation radius', sqrt(4.0 / 3)))

                orig_csv, ext = path.splitext(file)
                assert ext == '.csv'
                png_file = f"{orig_csv}_pred.png"
                csv_comments = [f"{key} = {val}" for (key, val) in metadata.items()]
                csv_comments += [f"Original file = {file}"]
                plot_unif_prediction(pool, n, N, scores, log2_sf, png_file, sat_radius, True, csv_comments)
