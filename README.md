# E: An exploration-and-exploitation adaptor for gradient-based optimziers (ALTO, an instance well-suited for large-batch training) 

## Features
-suitable for llm pretraining

## Installation
```
pip install git+https://github.com/zhaotong94/E
```
## Usage
Muon is an optimizer for the hidden weights of a neural network.
Other parameters, such as embeddings, classifier heads, and hidden gains/biases should be optimized using standard AdamW.
Muon should be used as follows:

```python
from E import E
origin_optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.90, 0.95), weight_decay=0.01)
optimizer = E(origin_optimizer)
```
```python
from ALTO import ALTO
origin_optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.90, 0.95), weight_decay=0.01)
```
You'll have to replace `model.body`, `model.head`, and `model.embed` with whatever is appropriate for your model.
E.g., for a ConvNet, you should use Muon to optimize all the convolutional filters except the first one, and AdamW to optimize everything else.
## Discussion on hyperparameter 
Typically, the default values of momentum (0.95), nesterov (True), and ns_steps (5) work well. Only the learning rate and weight decay must be tuned.
The learning rate should have built-in muP scaling: That is, as you scale up the model size, you shouldn't need to retune it.
## Citation
```
@inproceedings{
anonymous2025exploring,
title={Exploring Landscapes for Better Minima along Valleys},
author={Tong Zhao, Jiacheng Li, Yuanchang Zhou, Guangming Tan, Weile Jia},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=XxRKqFsvoK}
}
```
