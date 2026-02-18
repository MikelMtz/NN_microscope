import sys
import os
sys.path.append(os.path.abspath("src"))

import argparse
from ffnnmetrics.training import load_config, run_training

if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--config", type=str, default="/users/Mikel/Documents/00 - Universidades/3 - Oxford/Dissertation/NN_microscope_regul/configs/svhn6_grad_clip_annea_resnet.yaml")
    
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_training(cfg)
