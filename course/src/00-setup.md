---
title: Setup and how the course code works
summary: Install Python, OpenCV and friends in a virtual environment, download the code and data, and learn the two-flag convention every lesson script follows.
topics: venv · pip · OpenCV · project layout
---

## What you need

- **Python 3.10 or newer.** Check with `python3 --version`.
- About **200 MB** of disk space for the libraries and the data.
- Any editor. Jupyter works too: every script is organised in small blocks you can paste into cells.

## Get the code and the data

Download [cv-course-code.zip](cv-course-code.zip) and unzip it. You get this layout:

```text
cv-course/
├── cvutils.py              # tiny helpers: load data, draw/save figures
├── requirements.txt
├── 01_images_as_arrays.py  # one script per lesson
├── ...
├── 19_fingerprints.py
└── data/                   # every image and dataset used in the lessons
    ├── basics/  noise/  morphology/  color/  features/
    ├── calibration/        # chessboard pictures from my webcam
    ├── stereo/             # OpenCV's stereo calibration sample
    ├── modd/               # a stereo frame from the MODD maritime dataset
    ├── kitti/              # 15 frames of KITTI odometry, sequence 00
    ├── ct/                 # 21 CT slices (NIfTI)
    └── fingerprints/       # a fingerprint "database" and five crime-scene prints
```

## Create an environment

Always work in a virtual environment, so the course libraries never fight with your system packages.

```bash
cd cv-course
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows the activation line is `.venv\Scripts\activate`. The requirements are:

```text
--8<-- "requirements.txt"
```

!!! note "opencv-python vs opencv-contrib-python"
    `opencv-python` already contains SIFT, ORB, stereo matching and calibration. Install `opencv-contrib-python` *instead* (never both) only if you want the extra modules (`cv2.ximgproc`, `cv2.aruco` extras…). On a server without a screen, use `opencv-python-headless`.

Check that everything imports:

```bash
python -c "import cv2, numpy, skimage; print(cv2.__version__, numpy.__version__)"
```

## How every lesson script works

Each lesson is a plain script that you run from the `cv-course` folder:

```bash
python 05_noise_filtering.py            # opens each figure in a window
python 05_noise_filtering.py --save out # writes the figures to out/ instead
```

Numbers (errors, counts, timings) are printed to the terminal, and the lesson pages show the same output. The helpers live in `cvutils.py`; the only one you need to know is `U.show(name, images, titles)`, which draws a grid of images:

```python
--8<-- "cvutils.py:show"
```

Notice two details that will come back in lesson 01: colour images are converted from **BGR** (OpenCV's order) to RGB before drawing, and `uint8` grayscale images are drawn with `vmin=0, vmax=255`, so Matplotlib never secretly stretches the contrast.

!!! warning "Run scripts from the course folder"
    Data paths are resolved relative to `cvutils.py`, so the scripts work from anywhere, but `import cvutils` needs the folder on Python's path. The simplest is to `cd cv-course` first.

## Where the material comes from

The exercises follow the lab sessions of *Visão Robótica* (Robotic Vision, UFRN) and *TIM — Traitement d'Images* (INSA Rouen Normandie), my solutions on [GitHub](https://github.com/henriquecorneau/visao_robotica), and the ORB project I did at INSA. The text, figures and code here were rewritten for this course. Datasets: [KITTI odometry](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) (CC BY-NC-SA 3.0), [MODD](https://www.vicos.si/resources/modd/) (Kristan et al.), OpenCV's sample stereo images, the classic *cameraman* test image and the other test pictures used in the UFRN and INSA labs (including photos of Rouen and the low-contrast *Hawke's Bay* image from Wikimedia Commons).
