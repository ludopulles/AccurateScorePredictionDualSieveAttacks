#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <cstring>

typedef double FT;
constexpr FT two_pi = 2 * M_PI;

int num_vectors, dimension, *database = NULL;

extern "C" {
	void clean_up();
	void init_database(int _num_vectors, int _dimension, int *encoded);
	void bdd_scores(int num_samples, int seed, FT target_length, FT *output_table);
}

/*
 * Clears the database of vectors.
 */
void clean_up()
{
	if (database != NULL) {
		delete[] database;
		database = NULL;
	}
}

/*
 * Initializes the database of vectors.
 */
void init_database(int _num_vectors, int _dimension, int *encoded)
{
	num_vectors = _num_vectors;
	dimension = _dimension;

	// Make a copy of the database
	clean_up();
	database = new int[num_vectors * dimension];
	memcpy(database, encoded, num_vectors * dimension * sizeof(int));
}

/*
 * Gather scores of BDD targets.
 */
void bdd_scores(int num_samples, int seed, FT target_length, FT *output_table)
{
	assert(database != NULL);

    std::mt19937 gen{(unsigned int)seed};
    std::normal_distribution<FT> d{0, target_length / sqrt(dimension)};

	std::fill_n(output_table, 3 * num_samples, 0.0);

	FT *target = new FT[dimension];
	for (int i = 0; i < num_samples; i++) {
		FT norm_sq = 0;
		for (int j = 0; j < dimension; j++) {
			target[j] = d(gen);
			norm_sq += target[j] * target[j];
		}
		FT x0 = d(gen), x1 = d(gen), ball_norm_sq = norm_sq + x0 * x0 + x1 * x1;

		for (int k = 0; k < num_vectors; k++) {
			FT dot = 0.0;
			for (int j = 0; j < dimension; j++) {
				dot += target[j] * database[k * dimension + j];
			}

			// Sample from a sphere of radius `target_length`.
			output_table[i] += cos(two_pi * dot / sqrt(norm_sq) * target_length);

			// Sample from a ball of radius `target_length`.
			// To sample uniformly from a ball, sample from the uniform distribution on a sphere
			// and drop two coordinates.
			output_table[num_samples + i] += cos(two_pi * dot / sqrt(ball_norm_sq) * target_length);

			// Gaussian sample with mean 0 and standard deviation `target_length / sqrt(n)`.
			output_table[2 * num_samples + i] += cos(two_pi * dot);
		}
	}

	delete[] target;
}
