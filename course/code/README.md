# Computer Vision, from Pixels to 3-D — course code

Companion code for the course in the `course/` section of my portfolio site (one script per lesson).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python 05_noise_filtering.py            # show the figures
python 05_noise_filtering.py --save out # or save them to out/
```

Lesson 17 (ORB-SLAM3) has no script: it runs in Docker + ROS 2, see the lesson page.

Data: KITTI odometry (CC BY-NC-SA 3.0), MODD (Kristan et al.), OpenCV stereo samples,
and the test images of the UFRN *Visão Robótica* and INSA Rouen *TIM* labs.
