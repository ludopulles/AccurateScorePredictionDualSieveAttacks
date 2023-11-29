# Accurate Score Prediction for Dual-Sieve Attacks
This repository contains:

* `code/run.sh`: Script to generate all the data, and should generate all of the predictions.
* `code/bdd_sample.py`: sample score distribution for BDD targets of 3 distributions: uniform in sphere, uniform in ball, gaussian. This is done for a radius of `gh_factor * GH(n)`, where `gh_factor = 0.1, 0.2, ..., 1.0`.
* `code/bdd_predict.py`: give predictions for BDD targets. Call this script with the prediction dimension(s) as the command line argument(s). If for a particular dimension `n` and `gh_factor` the file `data/bdd_samples/bdd_scores_n{n}_ghf{gh_factor}.csv` exists, it will extract the experimental data from it, and combine experiments and prediction into one csv/png.
* `code/unif_sample.py`: sample score distribution for targets uniform modulo the lattice. This requires many parameters:
    * `n`: dimension
    * `lgT`: log (base 2) of the number of target samples to take
    * `fft`: dimension of FFT guessing step (requirement: fft <= lgT)
    * `satrat`: Saturation ratio of the dual sieve
    * `radius`: Saturation radius of the dual sieve
    * `q`: Prime used in generating random `q`-ary lattice
    * By supplying `--double_db`, you will use twice the number of dual vector pairs than expected.
    * By supplying `--exptime`, it only estimates the duration of the script, but does not execute it.
    * By supplying `-v`, the output is more verbose.
* `code/unif_predict.py`: plot prediction for uniform targets as csv-file and png-file. Two options for each command line argument:
    * supplying an integer, will interpret that number as a dimension, after which it will make a prediction in that dimension using default saturation ratio & saturation radius.
    * supplying a file, will make a prediction using all the parameters as listed in that csv file, containing the output of some call to `code/unif_sample.py`. Output is written to `{file}_pred.[csv|png]`.

## Requirements
* Preferably, a UNIX machine. The code is not tested on windows. In particular, you may perhaps have to change `dual_utils.so` to `dual_utils.dll` or so.
* C compiler, preferably `gcc`
* Python3
* [G6K](https://github.com/fplll/g6k/) which can be installed by running `pip install g6k`.
