# NN_microscope

A comprehensive framework for analyzing neural network training dynamics through kernel methods and feature-space metrics.

## Table of Contents
- [GitHub Setup](#github-setup)
- [Installation](#installation)
- [Quick Start: Training on MNIST](#quick-start-training-on-mnist)
- [Generating Plots](#generating-plots)
- [Available Datasets](#available-datasets)

## GitHub Setup

### Initial Setup

1. **Clone the repository:**
```bash
git clone https://github.com/NiclasGoering/NN_microscope.git
cd NN_microscope
```

## Installation

```bash
# Install dependencies (adjust based on your environment)
pip install torch torchvision numpy pandas matplotlib pyyaml
```

## Quick Start: Training on MNIST

### Step 1: Create or Edit Config File

Edit `configs/mnist_example.yaml` or create your own config


### Step 2: Run Training

**Exact command:**
```bash
python scripts/train.py --config configs/mnist_example.yaml
```



### Step 3: Monitor Training

Results will be saved to:
- `results/mnist_example/metrics.csv` - Training/test MSE
- `results/mnist_example/kernel/` - All kernel metrics

## Generating Plots

After training, generate all plots:

```bash
python scripts/plot_all.py --run mnist_example --outdir plots/mnist_example
```

Or using the results directory directly:
```bash
python scripts/plot_all.py --run results/mnist_example --outdir plots/mnist_example
```

## Available Datasets

The framework supports multiple datasets:

- **`staircase`**: Synthetic staircase functions (default)
- **`product_tree`**: Product tree functions
- **`mnist`**: MNIST digit classification (as regression)
- **`rotmnist`**: Rotated MNIST

Set `dataset: <name>` in your config file to use a specific dataset.

## Example Configurations

### MNIST (Full Dataset)
```yaml
dataset: mnist
train_size: 60000
test_size: 10000
dim: 784
```

### Rotated MNIST
```yaml
dataset: rotmnist
train_size: 60000
test_size: 10000
dim: 784
```

### Staircase (Synthetic)
```yaml
dataset: staircase
spec: "{8}+{1,2,3}"
dim: 50
train_size: 80000
test_size: 5000
```

### Product Tree
```yaml
dataset: product_tree
spec: "tree(k=32, r=2, d=5, inputs=uniform)"
dim: 50
train_size: 80000
test_size: 5000
```

## Output Structure

After training, your results directory will contain:

```
results/<run_name>/
├── metrics.csv                    # Training metrics
└── kernel/
    ├── summary.csv                # Kernel scalars
    ├── shares.csv                 # Layer shares
    ├── rotation_act.csv           # Forward rotation speeds
    ├── rotation_bp.csv            # Backward rotation speeds
    ├── alignment.csv              # Per-layer alignments
    ├── lastlayer_advanced.csv     # LARD, GSI, RPE, etc.
    ├── interlayer_ilard.csv       # Inter-layer metrics
    ├── interlayer_depth_profiles.csv
    ├── transported_alignment.csv
    └── *.npy files                # Eigenvalues, transfers, etc.
```

For a complete list of all metrics, see [METRICS_DOCUMENTATION.md](METRICS_DOCUMENTATION.md).

