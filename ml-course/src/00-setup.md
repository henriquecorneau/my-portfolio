---
title: Setup and how the course code works
summary: Install Python, scikit-learn and PyTorch in a virtual environment, download the code, and learn the conventions every lesson script follows — seeds, configuration objects, shapes in comments, and datasets that download themselves.
topics: venv · pip · PyTorch CPU/GPU · project layout · reproducibility
script: 00_overview.py
---

## What you need

- **Python 3.10 or newer.** Check with `python3 --version`.
- About **3 GB** of disk space: PyTorch is the big one, and the datasets downloaded during the course take about 600 MB.
- **No GPU required.** Every lesson runs on a laptop CPU. Lessons 13–16 train neural networks and take from a few minutes to ~15 minutes on a CPU; with a CUDA GPU they run in seconds and the code uses it automatically. The one exception is the Stable Diffusion script of lesson 17, which needs a GPU (a free Google Colab session works).

## Get the code

Download [ml-course-code.zip](ml-course-code.zip) and unzip it:

```text
ml-course/
├── mlutils.py                 # helpers: seeds, downloads, figures, plot_decision, train_classifier
├── requirements.txt
├── 01_knn_and_data_splits.py  # one script per lesson
├── ...
├── 17_diffusion.py
├── 17b_stable_diffusion.py    # needs a GPU
└── data/
    ├── mixtures/              # the INSA "mixture" dataset (2-D, two classes)
    ├── houses.csv             # 18 house coordinates (optimisation lab)
    ├── rouen.jpg              # a photo for clustering and detection
    ├── skate/                 # Skate detection dataset (YOLO format, CC BY 4.0)
    └── cache/                 # created on first run: downloaded public datasets
```

Small files ship with the code; public datasets (UCI wine quality and spambase, LIBSVM splice, MNIST, Fashion-MNIST, California housing, forest cover types, EuroSAT, LFW faces…) are downloaded the first time a lesson needs them, into `data/cache/`, and reused afterwards.

## Create an environment

```bash
cd ml-course
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

```text
--8<-- "requirements.txt"
```

!!! note "PyTorch: CPU or GPU build"
    On Linux, `pip install torch` installs a CUDA build of about 2 GB even on machines without a GPU. If you don't have one, install the much smaller CPU build first: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`, then the rest of the requirements. On a GPU machine, follow the selector on [pytorch.org](https://pytorch.org/get-started/locally/) to match your CUDA driver.

Check that everything imports:

```bash
python -c "import sklearn, torch; print(sklearn.__version__, torch.__version__, torch.cuda.is_available())"
```

## Your first run

`00_overview.py` trains five of the model families you will build in this course on the same toy problem — two interleaved half-moons with noise — and draws what each one learnt:

```python
--8<-- "00_overview.py:overview"
```

```text
logistic regression (04)   test accuracy 0.850
k-NN, k = 15 (01)          test accuracy 0.890
RBF SVM (05)               test accuracy 0.887
gradient boosting (08)     test accuracy 0.890
neural network (11)        test accuracy 0.897
```

!fig 00-hero | Predicted probability of the red class for each model (the black line is the 0.5 boundary). The numbers in brackets are the lessons where each one is built from scratch.

Four very different models end up within one point of accuracy of each other: on this data the noise, not the model, limits what is possible. Only the straight line of logistic regression is clearly too simple. Learning *why* each boundary has its shape — the line, the jagged neighbourhoods, the smooth kernel bumps, the axis-aligned staircase, the piecewise-linear folds — is what the course is about.

## How every lesson script works

```bash
python 05_svm.py            # opens each figure in a window
python 05_svm.py --save out # writes the figures to out/ instead
```

Numbers are printed to the terminal, and the lesson pages quote that output. The code on the pages is pulled directly from the scripts, so what you read is what runs.

### Same numbers every time

Machine learning code is full of randomness: data splits, initial weights, mini-batch order, dropout. Every script starts with `U.setup("05")`, which fixes the seeds of Python, NumPy and PyTorch:

```python
--8<-- "mlutils.py:seed"
```

Runs are then reproducible on the same machine and library versions. (Bit-exact reproducibility across different hardware is not guaranteed — GPU kernels may sum in a different order — so expect your numbers to differ slightly from the pages in the last digits.)

### Three habits used throughout

- **Hyper-parameters live in one place** — a `Config` dataclass or the arguments of one function call at the top of the experiment — never as magic numbers scattered through the code.
- **Shapes, dtypes and devices are written next to tensors** as comments: `# (B, 3, 64, 64) float32`. Most deep learning bugs are shape bugs; the comment makes them visible.
- **Every "from scratch" implementation is checked** against the library version (scikit-learn, PyTorch, torchvision) or against a numerical derivative before being trusted.

The figure helper you will meet most often draws the decision regions of any 2-D classifier:

```python
--8<-- "mlutils.py:decision"
```

## Where the material comes from

The lessons follow two courses I took and the projects around them: **Introduction to Machine Learning** at INSA Rouen Normandie (France) — k-NN, logistic regression, SVMs, model selection, trees and forests, boosting, clustering, PCA, optimisation and an introduction to deep learning — and **Deep Learning** at UFRN (Brazil) — neural network fundamentals, CNNs, YOLO object detection and generative AI. I rewrote everything in English and PyTorch, re-ran all of it, and kept the mistakes I made the first time as "bug" boxes, because they are the most useful part.

Datasets: [Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality) and [Spambase](https://archive.ics.uci.edu/dataset/94/spambase) (UCI), [splice](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/) (LIBSVM), the AIS athletes data (R package DAAG), MNIST, [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) (MIT), California housing and forest cover types (via scikit-learn), [EuroSAT](https://github.com/phelber/EuroSAT) (Sentinel-2 imagery, MIT), [Labeled Faces in the Wild](https://vis-www.cs.umass.edu/lfw/), and the [Skate detection](https://universe.roboflow.com/ufrnprojects/skate-detection-dlbga) dataset (CC BY 4.0) I labelled for the UFRN YOLO assignment.
