---
title: Machine Learning, from Linear Models to Deep Vision
summary: A hands-on course that starts with a nearest-neighbour classifier and ends with a diffusion model generating data from noise. Every algorithm is first written from scratch in NumPy or PyTorch, checked against the library you would use in practice, and run on real data — and every number and figure on these pages is the actual output of the code.
---

## About this course

This course grew out of two university courses I took — **Introduction to Machine Learning** at INSA Rouen Normandie (France) and **Deep Learning** at UFRN (Brazil) — and the deep learning I use in my research on autonomous drones and boats. I rewrote all of it as one path, in English, moved the deep learning to PyTorch, re-ran every experiment, and fixed the mistakes I made the first time. The mistakes are still there, in red boxes: un-normalised inputs that cost 50 points of accuracy, a model tuned on its own test set, an optimiser comparison where the optimiser never changed. They teach more than the parts that worked.

Each lesson follows the same rhythm:

1. **The idea** — the problem, and the maths behind it, kept short.
2. **From scratch** — an implementation you can read line by line: k-NN, gradient descent, an SVM dual, CART, AdaBoost, k-means, PCA, backpropagation, non-maximum suppression, ArcFace, a diffusion model…
3. **The library way** — the scikit-learn, PyTorch or Ultralytics call you would use in a real project, and a check that both agree.
4. **Run it** — every lesson is a single script. Run it, change a hyper-parameter, run it again.
5. **Pitfalls** — the bugs I actually hit, so you don't have to.

## Who it is for

Students and engineers who know Python and some linear algebra and want to *understand* machine learning instead of copying `model.fit()`. Parts I–III need high-school calculus (a derivative, the chain rule). Part IV builds neural networks from the ground up, and Part V applies them to vision. If you are new to images, the [Computer Vision course](../course/index.html) is the natural companion: it covers the classical side — filters, features, geometry — that this course builds on in lessons 13–16.

## How to follow it

Start with [lesson 00](00-setup.html) to install the environment and download the code. Parts I–III can be read in order or picked à la carte; lesson 06 (model selection) is the one not to skip. Part IV builds up in order (11 → 12 → 13 → 14), and each lesson of Part V stands on its own once Part IV is done.

!!! tip "Where this leads"
    Lessons 13–16 are the toolbox behind my projects: [boat perception](../projects/boat-vision.html) with a custom YOLOv11 detector for an autonomous surface vessel, [drone target tracking](../projects/tello-tracking.html) with a custom YOLOv8 model, and [monocular depth on a drone](../projects/tello-depth.html) with pretrained networks.
