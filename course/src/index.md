---
title: Computer Vision, from Pixels to 3-D
summary: A hands-on course that starts with a pixel and ends with a 3-D reconstruction of a street. Every method is first written from scratch in NumPy, then compared with the OpenCV function you would use in practice — and every figure on these pages is the real output of the code.
---

## About this course

This course grew out of two university courses I took and the projects I built around them: **Robotic Vision** in the mechatronics graduate program at UFRN (Brazil) and **Image Processing (TIM)** at INSA Rouen Normandie (France), plus the computer vision I use in my research on autonomous drones and boats. I rewrote all of it as one path, in English, cleaned up the code, fixed the mistakes I made the first time, and added the explanations I wish I had had.

Each lesson follows the same rhythm:

1. **The idea** — what problem we are solving and the maths behind it, kept as short as possible.
2. **From scratch** — an implementation in plain Python/NumPy that you can read line by line.
3. **The OpenCV way** — the one-liner you would use in a real project, and a check that both agree.
4. **Run it** — every lesson is a single script. Run it, change a parameter, run it again.
5. **Pitfalls** — the bugs I actually hit, so you don't have to.

## Who it is for

Students and engineers who know some Python and want to *understand* computer vision instead of copying function calls. You need high-school maths; a bit of linear algebra (matrices, vectors) helps from Part I on and becomes essential in Part V. No deep learning is required: this is the classical toolbox that every vision pipeline — including the neural ones — still relies on for calibration, geometry, pre- and post-processing. The learning side — from k-NN and SVMs to CNNs, YOLO and diffusion models — is in the companion [Machine Learning course](../ml-course/index.html).

## How to follow it

Start with [lesson 00](00-setup.html) to install the environment and download the code and data (one zip, about 20 MB). Parts I to IV build on each other; Part V (geometry and 3-D) needs lessons 11–14; the two case studies in Part VI reuse a bit of everything.

!!! tip "Where this leads"
    Lessons 15–17 are the foundations of two of my projects: the [boat perception pipeline](../projects/boat-vision.html) (stereo depth + GPS georeferencing) and [drone localisation in a pre-built map](../projects/drone-map-localization.html) with ORB-SLAM3.
