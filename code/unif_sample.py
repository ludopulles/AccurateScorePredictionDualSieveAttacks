#!/usr/bin/python3
"""
Code that executes a dual attack on a small dimension LWE instance and
gathers some statistics about the likeliness the attack will work.
"""
import argparse
from copy import copy
import ctypes
from datetime import datetime, timedelta
from functools import partial
from math import ceil, erfc, log2, sqrt
from multiprocessing import cpu_count, Pool
from time import perf_counter, process_time
from random import randint

from numpy import array, concatenate, int32, int64, float64, linalg, zeros

# This code requires an installation of G6K and FPyLLL (pip installable).
# FPLLL, G6K imports
from fpylll import LLL
from g6k import SieverParams
from g6k.siever import Siever

# Local imports:
from utils import change_dual_basis, generate_LWE_lattice, progressive_sieve
from unif_predict import plot_unif_prediction

def extract_log2_sf(buckets, outliers, num_samples):
    """
    Returns (xs, ys) of log2(survival function) of the data provided in buckets
    and its (right) outliers.
    :param outliers: all x-values of points that are considered to be rare
    events and as such these are all part of the output
    :param buckets: x-values of points that are likely to happen frequently and
    as such are aggregated into the same bucket.
    :param num_samples: total number of samples that was originally there (also
    counting the events with a negative score).
    """
    xs, ys = [], []
    num_bigger = sum(buckets) + len(outliers)
    # Assume half of the points have a negative score.
    log_samples = log2(num_samples)
    for (score, num_occ) in enumerate(buckets):
        if num_bigger > 0:
            xs.append(score)
            ys.append(log2(num_bigger) - log_samples)
        num_bigger -= num_occ
    for score in outliers:
        xs.append(score)
        ys.append(log2(num_bigger) - log_samples)
        num_bigger -= 1
    return xs, ys


###############################################################################
if __name__ == '__main__':
    # Parse the command line arguments:
    default_satrad = sqrt(4.0 / 3)

    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--threads', type=int, default=cpu_count()//2,
                        help='Number of CPU cores to use')
    parser.add_argument('-n', type=int, default=40,
                        help='Dimension of the (q-ary) lattice')
    parser.add_argument('--lgT', type=int, default=30,
                        help='log2 of # random targets to take in total')
    parser.add_argument('--fft', type=int, default=17,
                        help='Dimension of the FFT guessing step')
    parser.add_argument('--satrat', type=float, default=0.9,
                        help='Saturation ratio of the dual sieve')
    parser.add_argument('--radius', type=float, default=default_satrad,
                        help='Saturation radius of the dual sieve')
    parser.add_argument('-q', type=int, default=3329,
                        help='Prime to use in the q-ary lattice')
    parser.add_argument('--doubledb', action='store_true',
                        help='Use twice the number of dual vectors we expect?')
    parser.add_argument('--exptime', action='store_true',
                        help='Only guess the duration of the FFT step but does not execute it.')
    parser.add_argument('-v', action='store_true', help='Verbose')

    args = parser.parse_args()
    assert args.lgT > args.fft
    threads = min(cpu_count(), args.threads)
    print(f"Using {threads} CPU cores.")

    begin_wall_time = perf_counter()
    begin_cpu_time = process_time()

    B = LLL.reduction(generate_LWE_lattice(args.n, args.n // 2, args.q)[1])
    B_inv = linalg.inv(array(list(B)))

    # Construct a sublattice by multiplying the first fft basis vectors by 2.
    Bprime = copy(B)
    for i in range(args.fft):
        for j in range(args.n):
            Bprime[i, j] *= 2

    timestamp = perf_counter()

    sat_radius = args.radius

    g6k_dual = Siever(Bprime, SieverParams(threads=threads, dual_mode=True))
    progressive_sieve(g6k_dual, 0, args.n, double_db=args.doubledb, saturation_ratio=args.satrat,
                      verbose=args.v, saturation_radius=sat_radius)

    # Initially, dual vectors are given by coefficients in terms of the (order reversed) dual basis
    # of Bprime.
    # Now U^{-1} Bprime = diag(22...211...1) B
    # Dual vectors are expressed as combinations of (diag(22...211...1)B)^{-T}
    with Pool(threads) as pool:
        dual_db = pool.map(partial(change_dual_basis, g6k_dual.M.UinvT), g6k_dual.itervalues())

    sieve_time = perf_counter() - begin_wall_time
    N = len(dual_db)
    print(f"Dual sieve in dimension {args.n} produces {N} dual vectors.")
    print(f"Sieving took: {timedelta(seconds=int(sieve_time))}.", flush=True)

    variance = N * 0.5
    num_samples = 2**args.lgT

    # Since we expect more outliers, take a small value here.
    BUCKET_SIZE = 1
    while num_samples * 0.5 * erfc(BUCKET_SIZE / sqrt(2 * variance)) >= 100:
        BUCKET_SIZE += 1
    MAX_NUMBER_OUTLIERS = 10**6

    buckets = zeros(BUCKET_SIZE, dtype=int64)
    outliers = []

    # Initialize the binding to unif_sample.c
    c_binding = ctypes.CDLL("./unif_sample.so")
    c_int_p = ctypes.POINTER(ctypes.c_int32)
    c_int64_p = ctypes.POINTER(ctypes.c_int64)
    c_float_p = ctypes.POINTER(ctypes.c_double)

    enc_dual_db = zeros(N * args.n, dtype=int32)
    for (i, dual_vector) in enumerate(dual_db):
        for j in range(args.n):
            enc_dual_db[i * args.n + j] = dual_vector[j]
    enc_B_inv = zeros(args.n**2, dtype=float64)
    for i in range(args.n):
        for j in range(args.n):
            enc_B_inv[i * args.n + j] = B_inv[i, j]

    c_binding.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                          c_int_p, c_float_p]
    c_binding.init_dual_database(N, args.n, args.fft, args.q,
                                 enc_dual_db.ctypes.data_as(c_int_p), enc_B_inv.ctypes.data_as(c_float_p))

    # When doing the call to FFT_scores:
    c_binding.argtypes = [ctypes.c_int, c_float_p, c_float_p, c_int64_p]

    def work(num_jobs):
        _buckets = zeros(BUCKET_SIZE, dtype=int64)
        _outliers = zeros(MAX_NUMBER_OUTLIERS, dtype=float64)
        seed = randint(0, 2**31 - 1)

        buckets_p = _buckets.ctypes.data_as(c_int64_p)
        outliers_p = _outliers.ctypes.data_as(c_float_p)

        # Call the C function
        result = c_binding.FFT_scores(seed, num_jobs, BUCKET_SIZE,
                                      buckets_p, outliers_p)
        assert result < MAX_NUMBER_OUTLIERS
        return _buckets, copy(_outliers[:result])

    jobs = 2**(args.lgT - args.fft)
    jobs = [jobs // threads + (1 if i < (jobs % threads) else 0) for i in range(threads)]
    num_tests = 2**((27 if args.lgT > 40 else 23) - args.fft) # e.g.: fft = 18 => 32

    with Pool(threads) as pool:
        # Warm up for the benchmark
        for (_, _) in pool.imap_unordered(work, [num_tests] * threads):
            pass

        # Run the benchmark
        start_bench = perf_counter()
        for (_, _) in pool.imap_unordered(work, [num_tests] * threads):
            pass
        mid_bench = perf_counter()
        for (_, _) in pool.imap_unordered(work, [2*num_tests] * threads):
            pass
        end_bench = perf_counter()
        # Only consider the extra time needed for doing twice the work:
        secs_per_jobs = (end_bench - 2*mid_bench + start_bench) / num_tests

        # Make a prediction on time it will take
        duration_guess = timedelta(seconds=int(ceil(jobs[0] * secs_per_jobs)))
        finish_datetime = datetime.now() + duration_guess
        print(f"Estimated duration of sampling 2^{args.lgT}x: {duration_guess}")
        print(f"Estimated finish time: {finish_datetime}", flush=True)

        if args.exptime:
            exit(1)

        # Run the experiment
        timestamp = perf_counter()
        for (_buckets, _outliers) in pool.imap_unordered(work, jobs):
            for i in range(BUCKET_SIZE):
                buckets[i] += _buckets[i]
            outliers.append(_outliers)

    outliers = concatenate(outliers)
    outliers.sort()

    c_binding.argtypes = []
    c_binding.clean_up()
    del c_binding

    duration = perf_counter() - timestamp
    print(f"Computing scores took: {timedelta(seconds=int(duration))}.")

    end_wall_time = perf_counter()
    end_cpu_time = process_time()

    wall_time = end_wall_time - begin_wall_time
    cpu_time = end_cpu_time - begin_cpu_time

    png_file = f"../data/unif_n={args.n}_lgT={args.lgT}_rad={sat_radius:5.3f}.png"
    metadata = list(vars(args).items()) + [
        ('N', N), ('walltime', round(wall_time, 3)), ('cputime', round(cpu_time, 3)),
        ('saturation radius', sat_radius),
    ]
    metadata = [f"{key} = {val}" for (key, val) in metadata]
    real_scores, real_log2_sf = extract_log2_sf(buckets, outliers, num_samples)
    with Pool(threads) as pool:
        plot_unif_prediction(pool, args.n, N, real_scores, real_log2_sf, png_file, sat_radius=sat_radius, write_csv=True,
                             csv_comments=metadata)
