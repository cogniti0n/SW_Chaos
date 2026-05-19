#!/bin/bash

for n in 100 200 300 400 500
do  
    echo "running experiment: n=$n"
    python -m supp.scripts.sweep_2D_optimized_supp --n "$n" --perturbation_batchs 20 --numsamples 16 --numpoints_beta 20 --numpoints_eirat 20
done
