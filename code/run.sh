#!/bin/bash
echo "" > ~/run.log

# Figure 1, BDD
( time python bdd_sample.py -n 90 -s 100000 ) |& tee -a ~/run.log
( time python bdd_predict.py 90 ) |& tee -a ~/run.log

# Figure 2, Uniform, default saturation radius of sqrt(4/3) = 1.1547
( time python unif_sample.py -n 40 --lgT 48 --fft 16 ) |& tee -a ~/run.log
( time python unif_sample.py -n 50 --lgT 48 --fft 16 ) |& tee -a ~/run.log
( time python unif_sample.py -n 60 --lgT 48 --fft 16 ) |& tee -a ~/run.log
( time python unif_sample.py -n 70 --lgT 48 --fft 17 ) |& tee -a ~/run.log

# Figure 3, Uniform, with different saturation radius
( time python unif_sample.py -n 50 --lgT 40 --fft 16 --rad 1.18 ) |& tee -a ~/run.log
( time python unif_sample.py -n 60 --lgT 40 --fft 16 --rad 1.18 ) |& tee -a ~/run.log
( time python unif_sample.py -n 70 --lgT 40 --fft 17 --rad 1.20 ) |& tee -a ~/run.log
( time python unif_sample.py -n 80 --lgT 40 --fft 18 --rad 1.20 ) |& tee -a ~/run.log

# ( time python unif_predict.py ) |& tee -a ~/run.log
