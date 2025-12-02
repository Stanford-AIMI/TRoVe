# TRoVe: Discovering Error-Inducing Static Feature Biases in Temporal Vision-Language Models 
[![arXiv](https://img.shields.io/badge/arXiv-2512.01048-b31b1b.svg?style=for-the-badge)](https://arxiv.org/abs/2512.01048)
[![License](https://img.shields.io/github/license/stanford-aimi/trove?style=for-the-badge)](LICENSE)

This repository contains the official PyTorch implementation for [TRoVe: Discovering Error-Inducing Static Feature Biases in Temporal Vision-Language Models](https://arxiv.org/abs/2512.01048) (NeurIPS 2025).

![Overview](assets/img.png "")

## <img align="center" width="35" height="35" src="https://www.emojiall.com/images/240/google/1fa8e.png"> What is TRoVe?

When making predictions on temporal understanding tasks, vision-language models (VLMs) often rely on *static feature biases*, such as background or object features, rather than dynamic visual changes. Static feature biases are a type of shortcut and can contribute to systematic prediction errors. We introduce **TRoVe**, which can analyze a temporal VLM and discover learned static feature biases that induce errors at test time.

For additional details, please refer to our [paper](https://arxiv.org/abs/2512.01048), our [demo notebook](https://github.com/Stanford-AIMI/TRoVe/tree/main/demo.ipynb), and the documentation below.

## ⚡️ Installation
Use the following commands to clone and install this repository. We recommend using Python 3.9.
```python
git clone https://github.com/Stanford-AIMI/TRoVe.git
cd trove
pip install -e .
```

## ⚙️ Discovering Static Feature Biases with TRoVe
Code for using TRoVe to discover learned static feature biases in VLMs is provided in ```src/trove```. 

For a detailed walkthrough on running TRoVe, please refer to our demo notebook ```demo.ipynb```.


## 📎 Citation
If you find this repository useful for your work, please cite the following paper:

```
@inproceedings{
varma2025trove,
title={{TR}oVe: Discovering Error-Inducing Static Feature Biases in Temporal Vision-Language Models},
author={Maya Varma and Jean-Benoit Delbrouck and Sophie Ostmeier and Akshay S Chaudhari and Curtis Langlotz},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=KOwhczyFpg}
}
```

