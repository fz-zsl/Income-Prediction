<br />

<p align="center">
  <h1 align="center">Adult Income Prediction based on Neural Network</h1>
  <p align="center">
    <a href="https://github.com/fz-zsl">Shengli Zhou (12212232)</a>
  </p>
</p>
## Installation

The required packages are included in `requirements.txt`, you can build the environment for running the code by executing the following command in the project folder:

```bash
pip install -r requirements.txt
```

## TL;DR

```bash
python train.py --dataset train
python inference.py --dataset train
```

Then you can find the answers in `data/testlabel.txt`.

## Dataset

This project is based on the Adult Census Income dataset, which can be downloaded from [kaggle](https://www.kaggle.com/datasets/uciml/adult-census-income).

For simplicity, we've placed the downloaded data in `data/full` folder. Another version of dataset (the same data but different way of splitting data) provided by the project in CS311 is placed in `data/train`.

## Quick Start

- For preprocessing the dataset:

```bash
cd data
python preprocess.py --dataset [train | full] --sep
cd ..
```

- For training the model:

```bash
python train.py --dataset [train | full]
```

- For evaluating the model (note that only `full` dataset is available here as we don't have the answers to the `train` dataset)

```bash
python evaluate.py --dataset full
```

- For making inference:

```bash
python inference.py --dataset [train | full]
```

Then, you can find the predicted labels in `data/testlabel.txt` (or `data/testlabel_full` if you use the `full` dataset), each line in the text file represents an answer predicted according to the given information.

## Checkpoints

The official checkpoints (weights) can be found in the `checkpoints` folder.