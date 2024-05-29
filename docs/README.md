<div align="center">
<img src="logo.png" width="150" height="auto">
</div>

<h4 align="center">A lightweight deep learning framework to rapidly prototype AI models.</h4>

<p align="center">
    <a href="https://badge.fury.io/js/electron-markdownify">
    <img src="https://github.com/webstorms/DevTorch/actions/workflows/tests.yml/badge.svg">
<img src="https://github.com/webstorms/DevTorch/actions/workflows/linting.yml/badge.svg">
  </a>
  <a href="https://badge.fury.io/js/electron-markdownify">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg">
  </a>
<a href="https://doi.org/10.5281/zenodo.11383797"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.11383797.svg" alt="DOI"></a>
</p>

## Install DevTorch

Quick installation from PyPI:

```bash
pip install devtorch
```

## DevTorch overview

PyTorch is an amazing deep learning library. However, it's power comes with writing much boilerplate code which slows quickly testing new ideas. DevTorch helps you develop your PyTorch models faster with less code by
- Automatically keeping track of hyperparameters
- Quickly launch model training (also supports K-fold)
- Simple model eval: just pass a function

## Simple copy-paste example

Below is a simple example of DevTorch in action, showcasing how to quickly define, train and evaluate a model on the MNSIT dataset. This copy-paste boilerplate-code should be sufficient for most applications.

```python
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import devtorch

# 1. =========== Define your model ===========
class ANNClassifier(devtorch.DevModel):

    def __init__(self, n_in, n_hidden, n_out):
        super().__init__()
        self.layer1 = nn.Linear(n_in, n_hidden, bias=False)
        self.layer2 = nn.Linear(n_hidden, n_out, bias=False)
        self.init_weight(self.layer1.weight, "glorot_uniform")
        self.init_weight(self.layer2.weight, "glorot_uniform")

    def forward(self, x):
        x = F.leaky_relu(self.layer1(x.flatten(1, 3)))
        return F.leaky_relu(self.layer2(x))

# 2. =========== Train your model ===========
model = ANNClassifier(784, 1000, 10)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST("../data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST("../data", train=False, download=True, transform=transform)

def loss(output, target):
    return F.cross_entropy(output, target.long())

trainer = devtorch.get_trainer(loss, model=model, train_dataset=train_dataset, n_epochs=8, batch_size=128, lr=0.001, device="cuda")
trainer.train()

# 3. =========== Evaluate your model ===========
def eval_metric(output, target):
    return (torch.max(output, 1)[1] == target).sum().cpu().item()

scores = devtorch.compute_metric(model, test_dataset, eval_metric, batch_size=256)
print(f"Accuracy = {torch.Tensor(scores).sum()/len(test_dataset)}")
```
```console
INFO:trainer:Completed epoch 0 with loss 124.63664297852665 in 7.6264s
INFO:trainer:Completed epoch 1 with loss 42.02237884118222 in 7.5964s
INFO:trainer:Completed epoch 2 with loss 24.776000619633123 in 7.5570s
INFO:trainer:Completed epoch 3 with loss 16.104854418314062 in 7.5636s
INFO:trainer:Completed epoch 4 with loss 12.41756428591907 in 7.5666s
INFO:trainer:Completed epoch 5 with loss 11.086629197117873 in 7.5661s
INFO:trainer:Completed epoch 6 with loss 10.519949007764808 in 7.5741s
INFO:trainer:Completed epoch 7 with loss 9.502970711946546 in 7.5690s
INFO:trainer:Completed epoch 8 with loss 7.102849260583753 in 7.5699s
INFO:trainer:Completed epoch 9 with loss 6.427233138925658 in 7.5687s
Accuracy = 0.979200005531311
```

## More examples
The following notebooks contain example code showing how to use DevTorch for specific tasks.
#### Model
- [Weight initialization](../notebooks/model/Weight%20initialization.ipynb)
- [Tracking custom hyperparams](../notebooks/model/Custom%20hyperparams.ipynb)

#### Training
- [L1/L2 weight regularization](../notebooks/train/L1-L2%20weight%20regularization.ipynb)
- [Using model checkpoints](../notebooks/train/Checkpoints.ipynb)
- [Keeping track of multiple losses](../notebooks/train/Tracking%20losses.ipynb)
- [Customizing training: LR scheduling and grad clipping ](../notebooks/train/LR%20scheduling%20and%20grad%20clipping.ipynb)

#### Setting hyperparameters
- [Validate using K-fold validation](../notebooks/val/K-fold%20validation.ipynb)

#### Evals
- [Loading logs, models and hyperparams](../notebooks/eval/Loading.ipynb)
- [Custom eval](../notebooks/eval/Custom%20eval.ipynb)
- [Loading multiple evals](../notebooks/eval/Loading%20multiple%20evals.ipynb)

## Citation
If you use DevTorch in your work, please cite it as follows:
```bibtex
@software{devtorch2024,
  author       = {Taylor, Luke},
  title        = {{DevTorch: A lightweight deep learning framework to rapidly prototype AI models}},
  month        = may,
  year         = 2024,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.11383797},
  url          = {https://doi.org/10.5281/zenodo.11383797}
}
```

## License
DevTorch has a MIT license, as found in the LICENSE file.