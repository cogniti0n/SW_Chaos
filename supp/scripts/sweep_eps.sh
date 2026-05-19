#!/bin/bash

for eps in 1e-3 0.03 0.05 0.1 0.2
do
    echo "running experiment: eps=$eps"
    python -m supp.scripts.sweep_beta_optimized_supp --eps "$eps" --perturbation_batchs 20 --numsamples 16 --numpoints 20
done
