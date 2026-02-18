#!/bin/bash

set -e

cd "$HOME/NN_microscope_regul"

# Add src directory to PYTHONPATH
export PYTHONPATH="$HOME/NN_microscope_regul/src:$PYTHONPATH"

python scripts/train.py --config configs/svhn6_grad_clip_annea.yaml
