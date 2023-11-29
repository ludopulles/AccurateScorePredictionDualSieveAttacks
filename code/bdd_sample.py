#!/usr/bin/python3
"""
Code that executes a dual attack on a small dimension LWE instance and
gathers some statistics about the likeliness the attack will work.
This code computes a lot of scores for BDD-samples at a specific distance rel. to GH.
"""
import argparse
import csv
import ctypes
from functools import partial
from math import sqrt
from multiprocessing import cpu_count, Pool
from time import perf_counter, process_time
from numpy import concatenate, int32, float64, zeros

# This code requires an installation of G6K and FPyLLL (pip installable).
# FPLLL, G6K imports
from g6k import SieverParams
from g6k.siever import Siever

# Local imports:
from utils import change_basis, generate_LWE_lattice, GH, progressive_sieve


def write_bdd_scores(filename, metadata, sphere_scores, ball_scores, gauss_scores):
    fieldnames = ['sphere-score', 'ball-score', 'gauss-score', 'metakey', 'metaval']
    with open(filename, 'w', newline='', encoding='utf8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for (key, val) in metadata:
            writer.writerow({'metakey': key, 'metaval': val})

        for i in range(len(gauss_scores)):
            writer.writerow({
                'sphere-score': round(sphere_scores[i], 3),
                'ball-score': round(ball_scores[i], 3),
                'gauss-score': round(gauss_scores[i], 3),
            })


###############################################################################
if __name__ == '__main__':
    # Parse the command line arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--threads', type=int, default=cpu_count()//2,
                        help='Number of CPU cores to use')
    parser.add_argument('-n', type=int, default=40, help='Dimension of the (q-ary) lattice')
    parser.add_argument('-s', '--samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('-q', type=int, default=3329, help='Prime to use in the q-ary lattice')

    args = parser.parse_args()
    n, num_samples, q = args.n, args.samples, args.q
    threads = min(cpu_count(), args.threads)
    assert n % 2 == 0

    print(f"Using {threads} CPU cores.")
    database = None

    begin_wall_time = perf_counter()
    # We get sieve vectors in the primal rather than dual and take errors in the dual which is
    # simpler than having errors in the primal and sieve in the dual to get a list of dual vectors.
    B = generate_LWE_lattice(n, n // 2, q)[1]

    g6k = Siever(B, SieverParams(threads=threads))
    progressive_sieve(g6k, 0, n, saturation_ratio=0.99, verbose=True)

    # Initially, dual vectors are given by coefficients in terms of the (order
    # reversed) dual basis of Bprime.
    # Now U^{-1} Bprime = diag(22...211...1) B
    # Dual vectors are expressed as combinations of (diag(22...211...1)B)^{-T}

    with Pool(threads) as pool:
        database = pool.map(partial(change_basis, g6k.M.B), g6k.itervalues())
    N = len(database)

    print(f"Lattice sieving took {perf_counter() - begin_wall_time:.2f} seconds.")

    for gh_factor in [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]:
        begin_wall_time, begin_cpu_time = perf_counter(), process_time()

        # The (dual) vectors are from a lattice with determinant q^{n/2} so the shortest vector has
        # length ~`GH(n) * sqrt(q)`.
        # Thus, our targets need to be of length `ghf * GH(n) / sqrt(q)`.
        target_length = gh_factor * GH(n) / sqrt(q)
        len_duals = (2 * N)**(1/n) * GH(n) * sqrt(q)

        # Initialize the binding to bdd_sample.cpp
        enc_db = zeros(N * args.n, dtype=int32)
        for (i, vector) in enumerate(database):
            for j in range(args.n):
                enc_db[i * args.n + j] = vector[j]
        c_binding = ctypes.CDLL("./bdd_sample.so")
        c_binding.init_database(N, args.n, enc_db.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))

        # Generate jobs
        jobs = [(i, num_samples // threads +
                (1 if i < (num_samples % threads) else 0)) for i in range(threads)]

        def generate_samples(kwargs):
            """
            Generate `num_runs` many BDD scores, using the C(++) binding.
            :param kwargs: arguments to pass on to the C function.
            :param num_runs: number of samples to run.
            """
            seed, num_runs = kwargs
            table = zeros(3 * num_runs, dtype=float64)
            table_p = table.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            c_binding.bdd_scores(num_runs, seed, ctypes.c_double(target_length), table_p)
            return table[:num_runs], table[num_runs:-num_runs], table[-num_runs:]

        # Run the experiment
        sphere_scores, ball_scores, gauss_scores = [], [], []
        with Pool(threads) as pool:
            for _sscores, _bscores, _gscores in pool.imap_unordered(generate_samples, jobs):
                sphere_scores.append(_sscores)
                ball_scores.append(_bscores)
                gauss_scores.append(_gscores)
        c_binding.clean_up()
        del c_binding

        sphere_scores = concatenate(sphere_scores)
        ball_scores = concatenate(ball_scores)
        gauss_scores = concatenate(gauss_scores)

        sphere_scores.sort()
        ball_scores.sort()
        gauss_scores.sort()

        end_wall_time, end_cpu_time = perf_counter(), process_time()
        wall_time, cpu_time = end_wall_time - begin_wall_time, end_cpu_time - begin_cpu_time
        print(f"Sampling scores took {wall_time} seconds.")

        filename = f"../data/bdd_samples/bdd_scores_n{n}_ghf{gh_factor}.csv"
        metadata = list(vars(args).items()) + [
            ('gh_factor', gh_factor), ('N', N),
            ('walltime', round(wall_time, 3)), ('cputime', round(cpu_time, 3))
        ]
        write_bdd_scores(filename, metadata, sphere_scores, ball_scores, gauss_scores)
