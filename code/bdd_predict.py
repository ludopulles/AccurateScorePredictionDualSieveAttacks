#!/usr/bin/python3
"""
Code that executes a dual attack on a small dimension LWE instance and
gathers some statistics about the likeliness the attack will work.
This code computes a lot of scores for BDD-samples at a specific distance rel. to GH.
"""
from bisect import bisect_left
from functools import partial
from multiprocessing import cpu_count, Pool
from os import path
from sys import argv, stdout
from time import perf_counter

from mpmath import erf, exp, floor, gamma, inf, pi, quad, sqrt
from numpy import linspace
import matplotlib.pyplot as plt

# Local imports:
from utils import ball_distribution, GH


def read_bdd_scores(filename):
    """
    Read the BDD scores from the csv generated by bdd_sample.py.
    """
    metadata, sphere_scores, ball_scores, gauss_scores = {}, [], [], []
    with open(filename, newline='') as csvfile:
        for line in csvfile:
            parts = line.split(',')
            if line.startswith('sphere-score'):
                continue
            elif line.startswith(',,,'):
                metadata[parts[3]] = parts[4]
                continue
            sphere_scores.append(float(parts[0]))
            ball_scores.append(float(parts[1]))
            gauss_scores.append(float(parts[2]))

    # should not be needed:
    sphere_scores.sort()
    ball_scores.sort()
    gauss_scores.sort()

    return metadata, sphere_scores, ball_scores, gauss_scores


def normal_CDF(value, var):
    """
    Return Pr[X <= value], when X is normally distributed with mean 0 and variance var.
    :param value: evaluation point
    :param var: variance of the normal distribution
    :returns: the probability of having X <= value.
    """
    return float(0.5 + 0.5 * erf(value / sqrt(2 * var)))


def predict_CDF(avg, var, xs):
    """
    Evaluate the CDF of a normal distribution at the supplied points.
    :param xs: list of x-values where to evaluate the CDF
    :param avg: the expected mean of the normal distribution
    :param var: the expected variance of the normal distribution
    :returns: a list of y-points
    """
    return [normal_CDF(x - avg, var) for x in xs]


def predict_sphere(n, N, r12, xs):
    """
    Predicts the CDF of score distribution at points ``xs`` when targets are sampled uniformly from
    a sphere.
    :param n: dimension of the sphere
    :param N: number of dual vectors
    :param r12: radius of ball in which the dual vectors lie
    :param xs: list of x-values where to evaluate the CDF
    :returns: a list of y-points
    """
    avg, var = ball_distribution(n, r12, N)
    return predict_CDF(avg, var, xs)


def predict_ball(n, N, r12, xs):
    """
    Predicts the CDF of score distribution at points ``xs`` when targets are sampled uniformly from
    a ball.
    :param n: dimension of the ball
    :param N: number of dual vectors
    :param r12: radius of ball in which the dual vectors lie
    :param xs: list of x-values where to evaluate the CDF
    :returns: a list of y-points
    """
    def integrand_ball(score, factor):
        avg, var = ball_distribution(n, r12 * factor, N)
        return normal_CDF(score - avg, var) * n * factor**(n - 1)

    return [float(quad(partial(integrand_ball, x), [0.001, 0.5, 1])) for x in xs]


def chi_pdf(x, k):
    """
    Return the PDF of the chi-distribution.
    Source: https://en.wikipedia.org/wiki/Chi_distribution
    :param x: evaluation point
    :param k: dimension of sample
    :returns: PDF of the k-dimensional chi distribution evaluated at x.
    """
    # assert x >= 0
    return x**(k-1) * exp(-x*x/2) / (2**(k/2-1) * gamma(k/2))


def predict_gauss(n, N, r12, xs):
    """
    Predicts the CDF of score distribution at points ``xs`` when targets are sampled from some
    gaussian distribution.
    :param n: dimension of the ball
    :param N: number of dual vectors
    :param r12: radius of ball in which the dual vectors lie
    :param xs: list of x-values where to evaluate the CDF
    :returns: a list of y-points
    """
    def integrand_gauss(score, factor):
        avg, var = ball_distribution(n, r12 * factor, N)
        return normal_CDF(score - avg, var) * sqrt(n) * chi_pdf(factor * sqrt(n), n)

    return [float(quad(partial(integrand_gauss, x), [0.1, 1, inf])) for x in xs]


def make_predictions(n, gh_factor):
    # Read scores
    filename = f"../data/bdd_samples/bdd_scores_n{n}_ghf{gh_factor}.csv"

    # Note: no correction factor because of the determinant is needed, because
    #   det(primal) * det(dual) = 1.
    len_target = gh_factor * GH(n)
    len_duals = sqrt(4.0 / 3) * GH(n)
    r12 = len_duals * len_target

    has_csv = path.exists(filename)
    print(f"File {filename} does " + ("indeed" if has_csv else "not") + " exist.")
    if has_csv:
        metadata, sphere_scores, ball_scores, gauss_scores = read_bdd_scores(filename)
        assert int(metadata['n']) == n
        N = int(metadata['N'])
        num_samples = int(metadata['samples'])

        # Write sphere predictions
        idx_l, idx_r = round(0.01 * num_samples), round(0.99 * num_samples)
        x_min = min(sphere_scores[idx_l], ball_scores[idx_l], gauss_scores[idx_l])
        x_max = max(sphere_scores[idx_r], ball_scores[idx_r], gauss_scores[idx_r])
    else:
        # Factor 0.5 to account for the symmetry, factor 0.5 for taking saturation ratio 50%.
        N = floor(0.5 * 0.5 * (4.0/3.0)**(n/2))

        pred_avg, _ = ball_distribution(n, r12, N)
        x_min, x_max = float(-pred_avg), float(2 * pred_avg)

    xs = list(linspace(x_min, x_max, 200))

    # "Naive" prediction, prior to this work.
    heur_avg, heur_var = N * exp(-2*pi**2 * r12**2 / n), N * 0.5  # var <= N/2 in [MATZOV22]
    heur_ys = predict_CDF(heur_avg, heur_var, xs)

    pred_sphere_ys = predict_sphere(n, N, r12, xs)
    t0, pred_ball_ys, t1 = perf_counter(), predict_ball(n, N, r12, xs), perf_counter()
    print(f"Predicting ball (n={n:3}, ghf={gh_factor:3.1f}) took {t1-t0:7.3f} seconds", flush=True)
    t0, pred_gauss_ys, t1 = perf_counter(), predict_gauss(n, N, r12, xs), perf_counter()
    print(f"Predicting gaussian (n={n:3}, ghf={gh_factor:3.1f}) took {t1-t0:7.3f} seconds", flush=True)

    if has_csv:
        real_sphere_ys = [bisect_left(sphere_scores, x) / num_samples for x in xs]
        real_ball_ys = [bisect_left(ball_scores, x) / num_samples for x in xs]
        real_gauss_ys = [bisect_left(gauss_scores, x) / num_samples for x in xs]

        with open(f"../data/bdd_data/predictions_n{n}_ghf{gh_factor}.csv", "w", encoding="ascii") as f:
            f.writelines([
                "score,CDF_heur,CDF_sphere_real,CDF_sphere_pred,CDF_ball_real,CDF_ball_pred,CDF_gauss_real,CDF_gauss_pred\n",
                f"# Number of samples used: {len(sphere_scores)}\n",
                f"# Number of dual vectors: {N}\n",
            ])

            for i, x in enumerate(xs):
                f.write(f"{x:.3f},{heur_ys[i]:.6f},{real_sphere_ys[i]:.6f},{pred_sphere_ys[i]:.6f},{real_ball_ys[i]:.6f},{pred_ball_ys[i]:.6f},{real_gauss_ys[i]:.6f},{pred_gauss_ys[i]:.6f}\n")

    fig = plt.figure()
    sub_plot = fig.add_subplot(1, 1, 1)
    sub_plot.set_xlim(x_min, x_max)

    sub_plot.plot(xs, heur_ys, color='purple', linestyle='dashed', label='heuristic')

    if has_csv:
        all_ys = [i/num_samples for i in range(num_samples)]
        sub_plot.plot(sphere_scores, all_ys, color='tomato', label='sphere BDD targets')
        sub_plot.plot(ball_scores, all_ys, color='lawngreen', label='ball BDD targets')
        sub_plot.plot(gauss_scores, all_ys, color='cyan', label='gaussian BDD targets')

    sub_plot.plot(xs, pred_sphere_ys, color='darkred', linestyle='dashed', label='sphere prediction')
    sub_plot.plot(xs, pred_ball_ys, color='darkgreen', linestyle='dashed', label='ball prediction')
    sub_plot.plot(xs, pred_gauss_ys, color='darkcyan', linestyle='dashed', label='gaussian prediction')

    sub_plot.legend(loc='lower right')
    plt.savefig(f"../data/bdd_data/plot_n={n}_ghf={gh_factor}.png", dpi=500)
    fig.clear()


if __name__ == '__main__':
    # Parse the command line arguments:
    if len(argv) <= 1:
        print(f"Usage: {argv[0]} [n ...]\n  - n: dimension of lattice.")

    ns, ghs = map(int, argv[1:]), [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    jobs = [(n, gh) for n in ns for gh in ghs]
    threads = min(len(jobs), cpu_count() // 2)

    with Pool(threads) as pool:
        pool.starmap(make_predictions, jobs)
