  <a href="https://www.python.org/"><img alt="Python Version" src="https://img.shields.io/badge/Python-%E2%89%A53.10-blue" /></a>
  <a href="https://pytorch.org/"><img alt="PyTorch Version" src="https://img.shields.io/badge/PyTorch-%E2%89%A52.0.0-green" /></a>
<!-- <div align="center">
</div> -->
--------------------------------------------------------------------------------
# Subtrajectory Evaluation Balance (Sub-EB) for GFlowNet
Code for the ICLR 2026 paper '[Evaluating GFlowNet from partial episodes for stable and flexible policy-based training](https://proceedings.mlr.press/v235/niu24c.html)'

## Installation
Clone the code and set ```./Sub-EB/gfn-subeb``` as the root working directory

Create environment with conda:
```
conda create -n gfn_env python==3.10
conda activate gfn_env

pip install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install -r requirement.txt
```


## Citation

If you find our code useful, please considering citing our paper in your publications. We provide a BibTeX entry below.


```bibtex
@inproceedings{niu2026evaluating,
title={Evaluating {GF}lowNet from partial episodes for stable and flexible policy-based training},
author={Puhua Niu and Shili Wu and Xiaoning Qian},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=kWM0etSpBG}
}
