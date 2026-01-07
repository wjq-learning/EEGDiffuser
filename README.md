<div align="center">

# EEGDiffuser

_Label-guided EEG signals synthesis via diffusion model for BCI applications_

[![Paper](https://img.shields.io/badge/Paper-Neurocomputing-008B8B)](https://www.sciencedirect.com/science/article/pii/S0925231226000330)
![GitHub Repo stars](https://img.shields.io/github/stars/wjq-learning/EEGDiffuser)

</div>

<p align="center">
    🔍&nbsp;<a href="#-about">About</a>
    | 🔨&nbsp;<a href="#-setup">Setup</a>
    | 🚢&nbsp;<a href="#-train">Train</a>
    | 🔗&nbsp;<a href="#-citation">Citation</a>
</p>

🔥 NEWS: The paper "_EEGDiffuser: Label-guided EEG signals synthesis via diffusion model for BCI applications_" has been accepted by Neurocomputing!

## 🔍 About

We propose **EEGDiffuser**, a **label-conditioned diffusion model** for generating EEG signals to alleviate data scarcity in BCI.

<div align="center">
<img src="figure/model.png" style="width:100%;" />
</div>


## 🔨 Setup

Install [Python](https://www.python.org/downloads/).

Install [PyTorch](https://pytorch.org/get-started/locally/).

Install other requirements:

```commandline
pip install -r requirements.txt
``` 

## 🚢 Train
You can train EEGDiffuser on our dataset or your custom dataset using the following code:
```commandline
python diff_main.py
```

## 🔗 Citation

If you're using this repository in your research or applications, please cite using the following BibTeX:

```bibtex
@article{wang2026eegdiffuser,
  title={EEGDiffuser: Label-guided EEG signals synthesis via diffusion model for BCI applications},
  author={Wang, Jiquan and Zhao, Sha and Luo, Zhiling and Zhou, Yangxuan and Li, Shijian and Pan, Gang},
  journal={Neurocomputing},
  pages={132636},
  year={2026},
  publisher={Elsevier}
}
```