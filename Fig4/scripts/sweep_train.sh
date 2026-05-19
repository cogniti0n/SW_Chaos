#!/bin/bash

for SEED in {1..10}
do

python -m Fig4.scripts.sweep_train_lorenz --seed "$SEED"
python -m Fig4.scripts.sweep_train_mnist --seed "$SEED"
python -m Fig4.scripts.sweep_train_remote_bandwidth --seed "$SEED"

done
