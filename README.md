# E: An exploration-and-exploitation adaptor for gradient-based optimizers (suitable for large-batch training) 

## Features
E is a gradient-based optimizer adaptor for rendering the origin optimizer

  -to explore (alpha<0) for better minima along valleys of landscapes after finding a minimum.
  
  -to exploit (alpha>0, converging fast) the minimum when the optimizer finds its neighbor, trying to find the bottom of the minimum.

It is very suitable for large-batch training (batch size >1K, the larger the better). 
In small batch case, the outperformances of adapted optimizers are usually marginal.
## Installation
```
pip install git+https://github.com/zhaotong94/E
```
or
```
pip install e-optimizer-adaptor
```
## Usage
Here, we instantiate EAdamW from AdamW for large-batch training and EAdam from Adam for small-batch training as examples:
```python
from E import E
origin_optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.99), weight_decay=1e-4)
optimizer = E(origin_optimizer, alpha=0.5, beta=0.01)
```
```python
from E import E
origin_optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, betas=(0.9, 0.99), weight_decay=1e-4)
optimizer = E(origin_optimizer, alpha=-5, beta=0.99)
```
For large-batch pre-training task, we recommand using ALTO directly (adapted Lamb, but the Lamb in origin paper is without bias correction).
```python
from ALTO import create_ALTO_optimizer
optimizer = create_ALTO_optimizer(model, lr=0.01, betas=(0.99, 0.9, 0.99), alpha=-5, weight_decay=1e-4, eps=1e-8)
```
## Discussion on hyperparameter 
The larger the batch size is, the larger the $-\alpha$ and $\beta_1$ (beta in E.py, the first element of betas in ALTO.py) should be. Hence, we set $\alpha=0.5, \beta_1=0.01$ in small batch training (batch size $<$1K) and $\alpha=-5, \beta_1=0.99$ in large batch case (batch size $\geq$1K), unless otherwise specified. If not mentioned, we set $\beta_2=0.9, \beta_3=0.99, \lambda=10^{-4}, \varepsilon_1=10^{-6}, \varepsilon_2=10^{-6}, \varepsilon_3=10^{-10}$. These parameters allow ALTO ample room for performance improvement. We only adjust $\beta_1$ and $\eta$ for ALTO, while for other optimizers, we tune all hyperparameters.
## Citation
```
@inproceedings{zhao2025exploring,
title={Exploring Landscapes for Better Minima along Valleys},
author={Tong Zhao, Jiacheng Li, Yuanchang Zhou, Guangming Tan, Weile Jia},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=XxRKqFsvoK}
}
```
