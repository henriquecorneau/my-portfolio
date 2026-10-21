# Machine Learning, from Linear Models to Deep Vision — course code

Companion code for the course in the `ml-course/` section of my portfolio site (one script per lesson).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python 05_svm.py            # show the figures
python 05_svm.py --save out # or save them to out/
```

Small datasets are in `data/`; larger public datasets (UCI, LIBSVM, MNIST, Fashion-MNIST,
CIFAR-10, LFW, covertype…) are downloaded on first use into `data/cache/`.
Lessons 13–17 train neural networks: they run on a CPU in a few minutes each, much faster on a GPU.
`17b_stable_diffusion.py` needs a GPU.

Data shipped here: the INSA "mixture" dataset, 18 house coordinates (INSA optimisation lab),
a photo of Rouen, and the Skate detection dataset (Roboflow Universe, CC BY 4.0) with the
training log of my original YOLOv8 run.
