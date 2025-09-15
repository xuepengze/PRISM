# PRISM: PROGRESSIVE RAIN REMOVAL WITH INTEGRATED STATE-SPACE MODELING

The code implementation of the PRISM model.

# Method

## Graphical Abstract
<img src="https://raw.githubusercontent.com/xuepengze/PRISM/main/docs/PRISM-1757930845711-6.png" alt="PRISM" style="zoom:98%;" />

<p align="center">
<img src="https://github.com/xuepengze/PRISM/blob/main/docs/HA-UNet.png?raw=true" width=52% height=35% class="center">
<img src="https://github.com/xuepengze/PRISM/blob/main/docs/HDMamba.png?raw=true" width=46% height=52% class="center">
</p>

# Get Started

## Environment

You can create a virtual environment for the following environment deployment.

```bash
CUDA 11.7
Python 3.8.20
PyTorch 1.13.1+cu117
```

install PyTorch 1.13 + CUDA 11.8

```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

```bash
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 -c pytorch -c nvidia
```

To use the selective scan， the `mamba_ssm` library is needed to install with the folllowing command. 

```bash
pip install causal_conv1d==1.0.0
pip install mamba_ssm==1.0.1
```

If the installation fails, we recommend visiting the official website to download the mamba environment compatible with your version and install it manually. 

```bash
https://github.com/Dao-AILab/causal-conv1d/releases
```

```bash
pip install causal_conv1d-1.0.0+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
```

```bash
pip install mamba_ssm-1.0.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
```

 Continue installing additional  python libraries

```bash
conda install --yes --file requirements.txt
```

## Dataset

We use **Mixed Training Track**, the Rain13K benchmark is used for training, while Test100, Rain100H, Rain100L, Test2800, and Test1200 are used for testing. Your file structure will be like:

```bash
├── train
│   ├── input
│   └── target
└── test
    ├── Rain100H
    │   ├── input
    │   └── target
    ├── Rain100L
    │   ├── input
    │   └── target
    ├── Test100
    │   ├── input
    │   └── target
    ├── Test1200
    │   ├── input
    │   └── target
    └── Test2800
        ├── input
        └── target
```

Modify your dataset path in train.py `Dataset_path  = 'YourDataset'` .

## Installation

Clone the repository:

```bash
git clone https://github.com/ xuepengze/PRISM.git
```

## Usage

```bash
# train
python train.py 
```

```bash
# test
python test.py
```

```bash
# Evaluating
python PSNR_SSIM.py --test_y_channel
```

# Contact

We are glad to hear from you. If you have any questions, please feel free to contact XPZ2291811798@Gmail.com.






