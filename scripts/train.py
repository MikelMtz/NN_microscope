import argparse, os
from ffnnmetrics.training import load_config, run_training

if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--config", type=str, default="/home/goring/NN_microscope/configs/default.yaml")
    
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_training(cfg)