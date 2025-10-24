# E: An exploration-and-exploitation adaptor for gradient-based optimizers (suitable for large-batch training) 

## Features
E is a gradient-based optimizer adaptor for rendering the origin optimizer
  - to explore (alpha<0) for better minima along valleys of landscapes after finding a minimum.
  - to exploit (alpha>0, converging fast) the minimum (trying to find the bottom) when the optimizer approximates its neighbor.

It is very suitable for large-batch training (batch size $\geq1K$, the larger the better). 

In small batch training case (batch size $<1K$), the advantages of the adapted optimizers are relatively limited.
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
The $\alpha$ and $\beta_1$ in paper correspond to the alpha and beta in `E.py` (alpha and the first element of betas in `ALTO.py`), respectively. The $\beta_2$ and $\beta_3$ in paper correspond to the second and third elements of betas in `ALTO.py`. Their default values are $\beta_2=0.9$ and $\beta_3=0.99$. 

Our following discussion focuses on $|\alpha|<\frac{1}{1-\beta_1}$ and $0\leq\beta_1<1$. Generally, the larger the batch size is, the larger the $-\alpha$ and $\beta_1$ should be. Hence, we default to $\alpha=0.5, \beta_1=0.01$, if batch size $<1K$. If batch size $\geq1K$ we default to $\alpha=-5$ and $\beta_1=0.99$.

In more depth, $\beta_1$ measures the persistence of exploration, while $|\alpha|$ determines the scale of local minima that can be escaped for exploring landscape. For large batche cases, we set $\alpha$ to be negative for larger exploration range, flatter minima, and this leads to a better test performance. However, for small batche cases we set it positive. Although a negative $\alpha$ suggests flatter local minima, the resulting improvement in generalization is insufficient to offset the usually lower training loss achieved with positive $\alpha$. If a flat minimum is desired, setting $\alpha$ to a negative value is also acceptable.
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
