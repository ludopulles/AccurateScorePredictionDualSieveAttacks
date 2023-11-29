"""
Helper functions
"""
import functools
from math import ceil, pi, sqrt
from mpmath import gamma, hyp0f1

try:
    from fpylll import IntegerMatrix
except ImportError:
    print("Warning: couldn't import FPyLLL or G6K")


@functools.cache
def GH(dim):
    """
    Returns the expected length of the shortest vector in a lattice of dimension d and volume 1.
    """
    return gamma(dim/2 + 1)**(1/dim) / sqrt(pi)
    # return sqrt(dim * (pi * dim)**(1.0/dim) / (2 * pi * exp(1)))


def average_score_in_sphere(n, r12):
    """
    Return E[exp(2pi i <v,w>)] where v is uniform from the unit n-sphere and w is of length `r12`.
    """
    # return gamma(n/2) * besselj(n/2 - 1, 2 * pi * r12) / (pi * r12)**(n/2 - 1)
    return hyp0f1(n/2, -(pi * r12)**2)


def average_score_in_ball(n, r12):
    """
    Return E[exp(2pi i <v,w>)] where v is uniform from the unit n-ball and w is of length `r12`.
    """
    # return average_score_in_sphere(n + 2, r12)
    # return gamma(n/2 + 1) * besselj(n/2, 2 * pi * r12) / (pi * r12)**(n/2)
    return hyp0f1(n/2 + 1, -(pi * r12)**2)


def ball_distribution(n, r12, multiplier=1):
    """
    Returns the distribution (mean and variance) of e^{2pi i <x, t>} where x is uniform in a ball
    of radius r1 and t is of length r2.
    :param n: dimension.
    :param r12: product of r1 and r2.
    :param multiplier: distribution when summing `multiplier` independent samples.
    :returns: average and variance
    """
    average1 = average_score_in_ball(n, r12)
    average2 = average_score_in_ball(n, r12 * 2)
    return (multiplier * average1, multiplier * (0.5 + 0.5 * average2 - average1**2))


def generate_LWE_lattice(m, n, q):
    """
    Generate a basis for a random `q`-ary latice with `n` secret coefficients and `m` samples,
    i.e., it generates a matrix B of the form

        I_n A
        0   q I_{m-n},

    where I_k is the k x k identity matrix and A is a n x (m-n) matrix with
    entries uniformly sampled from {0, 1, ..., q-1}.

    :param m: the dimension of the final lattice
    :param n: the number of secret coordinates.
    :param q: the modulus to use with LWE

    :returns: The matrix A and B from above
    """
    B = IntegerMatrix.random(m, "qary", k=m-n, q=q)
    A = B.submatrix(0, n, n, m)
    return A, B


def progressive_sieve(g6k, l, r, saturation_ratio=0.9, verbose=False, double_db=False, saturation_radius=None):
    """
    Sieve in [l, r) progressively. The g6k object will contain a list of short vectors.
    Taking l > 0, will cause G6K to sieve in a projected sublattice of the full basis.

    :param g6k: Siever object used for sieving
    :param l: integer indicating number of basis vectors to skip at the beginning.
    :param r: integer indicating up to where to sieve.
    :param saturation_ratio: ratio to pass on to G6K (i.e. ratio of lattice vectors in the ball of
    radius sqrt(4/3) that should be in the database).
    :param verbose: boolean indicating whether or not to output progress of sieving.
    """
    if verbose:
        print(f"Sieving [{max(l, r-40):3}, {r:3}]", end="", flush=True)
    g6k.initialize_local(l, max(l, r - 40), r)
    g6k(alg="gauss")
    while g6k.l > l:
        # Perform progressive sieving with the `extend_left` operation
        if verbose:
            print(f"\rSieving [{g6k.l:3}, {g6k.r:3}]...", end="", flush=True)
        g6k.extend_left()
        g6k("bgj1" if g6k.r - g6k.l >= 45 else "gauss")
    with g6k.temp_params(saturation_ratio=saturation_ratio, db_size_factor=6):
        g6k(alg="hk3")

    num_dual_vectors = ceil((1 if double_db else 0.5) * saturation_ratio * sqrt(4 / 3)**(r - l))
    if saturation_radius is not None and abs(saturation_radius - sqrt(4/3)) > 1e-6:
        num_dual_vectors = ceil((1 if double_db else 0.5) * saturation_ratio * saturation_radius**(r - l))
        # Note: G6K expects saturation radius passed as a **squared length**, i.e. default is 4/3.
        with g6k.temp_params(saturation_ratio=saturation_ratio, db_size_factor=6,
                             db_size_base=saturation_radius, saturation_radius=saturation_radius**2):
            g6k(alg="hk3")

    if verbose:
        print("\rSieving is complete! ", flush=True)
    # Number of dual vectors that is used in a full sieve is (4/3)^{n/2}.
    # Take into account 1/2 since G6K only saves exactly one of the vectors w or -w.
    if len(g6k) < num_dual_vectors:
        print(f"Not enough dual vectors found: {len(g6k)}, expected >= {num_dual_vectors}.")
    assert len(g6k) >= num_dual_vectors
    g6k.resize_db(num_dual_vectors)


def change_basis(basis, vector):
    """
    Puts a vector `vector` specified by coefficients into basis `basis`, i.e.  calculate
    `vector * basis`.
    :param basis: an IntegerMatrix containing the basis for a lattice.
    :param vector: a vector specifying a lattice vector by its coefficients (in terms of `basis`).
    :returns: the same lattice vector as `vector` but now expressed in the canonical basis.
    """
    return basis.multiply_left(vector)


def change_dual_basis(basis, vector):
    """
    Puts a *dual* vector `vector` specified by coefficients into basis `basis`, i.e. calculate
    `vector * reversed(basis)`, since the output of a sieve with dual mode on returns the
    coefficients in reversed order in G6K.
    :param basis: an IntegerMatrix containing the basis for a lattice.
    :param vector: a vector specifying a lattice vector by its coefficients (in terms of `basis`).
    :returns: the same lattice vector as `vector` but now expressed in the canonical basis.
    """
    return basis.multiply_left(list(reversed(vector)))
