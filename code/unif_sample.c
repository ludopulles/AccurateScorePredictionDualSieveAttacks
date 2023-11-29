#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define FT double

/*
 * Perform the Walsh-Hadamard Transform in time O(n 2^n) on `table` assumed to be of size 2^n.
*/
static inline void walsh_hadamard_transform(FT *table, const int n)
{
	FT x;

	// unroll level i=0:
	for (int j = 0; j < (1 << n); j += 2) {
		x = table[j|1];
		table[j|1] = table[j] - x;
		table[j] += x;
	}

	// unroll level i=1:
	for (int j = 0; j < (1 << n); j += 4) {
		x = table[j|2];
		table[j|2] = table[j] - x;
		table[j] += x;

		x = table[j|3];
		table[j|3] = table[j|1] - x;
		table[j|1] += x;
	}

	for (int i = 2, p2 = 1 << i; i < n; i++, p2 <<= 1) {
		for (int j = 0, nj = p2 << 1; j < (1 << n); j = nj, nj += 2*p2) {
			for (int l = j, r = j + p2; r < nj; l++, r++) {
				x = table[r];
				table[r] = table[l] - x;
				table[l] += x;
			}
		}
	}

	/* for (int i = 0; i < n; i++) {
		for (int j = 0; j < (1 << n); ) {
			if ((j >> i) & 1) { j++; k++; continue; }

			x = table[j], y = table[k];
			table[j++] = x + y;
			table[k++] = x - y;
		}
	} */
}


int num_dual_vectors, dimension, dimension_fft, modulus;
int *dual_db = NULL, *indices = NULL;
FT *basis_inverse = NULL;

/*
 * Clear the database of dual vectors.
 */
void clean_up()
{
	if (dual_db != NULL) {
		free(dual_db);
	}
	if (indices != NULL) {
		free(indices);
	}
	if (basis_inverse != NULL) {
		free(basis_inverse);
	}
}

/*
 * Initialize the database of dual vectors.
 */
void init_dual_database(int _num_dual_vectors, int _dimension, int _dimension_fft, int _modulus,
		int32_t *_dual_db, FT *_basis_inverse)
{
	num_dual_vectors = _num_dual_vectors;
	dimension = _dimension;
	dimension_fft = _dimension_fft;
	modulus = _modulus;

	// Make a copy of the database
	clean_up();

	// Read the dual database:
	dual_db = (int*) calloc(num_dual_vectors * dimension, sizeof(int));
	memcpy(dual_db, _dual_db, num_dual_vectors * dimension * sizeof(int));

	// Determine all the indices
	indices = (int*) calloc(num_dual_vectors, sizeof(int));
	for (int i = 0; i < num_dual_vectors; i++) {
		indices[i] = 0;
		for (int j = 0; j < dimension_fft; j++) {
			indices[i] |= (dual_db[i * dimension + j] & 1) << j;
		}
	}

	// Read the basis inverse
	basis_inverse = (FT*) calloc(dimension * dimension, sizeof(FT));
	memcpy(basis_inverse, _basis_inverse, dimension * dimension * sizeof(FT));
}

/*
 * Generate `iterations * 2^{dimension_fft}` many scores, and put them:
 * - in outliers, if the score is above `threshold`, or else
 * - in buckets, if the score is still positive, and otherwise,
 * - discard the score (when it is negative).
 *
 * Targets are generated uniformly in (Z/modulusZ)^{dimension}, using a PRNG seeded with the supplied `seed`.
 */
int FFT_scores(int seed, int iterations, int threshold,
		int64_t *buckets, FT *outliers)
{
	size_t table_size = (size_t)1 << dimension_fft, num_outliers = 0;
	FT *target = (FT*) calloc(dimension, sizeof(FT));
	FT *FFT_table = (FT*) calloc(table_size, sizeof(FT));
	assert(dual_db != NULL);

	// Set a seed:
	srand(seed);

	for (int iter = 0; iter < iterations; iter++) {
		// Initialize the table with zeros.
		memset(FFT_table, (int) 0, table_size * sizeof(FT));

		// Generate a target, uniformly in (Z/qZ)^n, expressed in basis B.
		for (int i = 0; i < dimension; i++) {
			target[i] = 0.0;
		}
		for (int i = 0; i < dimension; i++) {
			int modq = rand() % modulus;
			for (int j = 0; j < dimension; j++) {
				target[j] += modq * basis_inverse[i*dimension + j];
			}
		}

		// Initialize the database.
		for (int i = 0, k0 = 0; i < num_dual_vectors; i++) {
			FT dot_product = 0.0;
			for (int j = 0; j < dimension; j++) {
				dot_product += target[j] * dual_db[k0++];
			}
			FFT_table[indices[i]] += cos(2 * M_PI * dot_product);
		}

		// Perform the Walsh-Hadamard Transform.
		walsh_hadamard_transform(FFT_table, dimension_fft);

		for (size_t i = 0; i < table_size; i++) {
			int score = lround(FFT_table[i]);
			// Remove negative scores, we are not interested in these.
			if (score >= 0) {
				if (score < threshold) {
					// Place this score in a bucket
					buckets[score]++;
				} else {
					// Place this score in an outlier, it's rare.
					outliers[num_outliers++] = FFT_table[i];
				}
			}
		}
	}

	return num_outliers;
}
