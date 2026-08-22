# Beyond Foundation Models: Distilling Geometric Priors for Lightweight Monocular Depth Estimation in Endoscopy
IEEE Transactions on Medical Imaging, 2026


## Overview

<p align="center">
  <img src="assets/overall-new.png" width="95%">
</p>



## Environment

Please create the environment using:

```bash
conda env create -f environment.yml
conda activate *

## Evaluation

```bash
python evaluate_depth.py --data_path ./your_data --load_weights_folder ./your_weight --eval_mono
