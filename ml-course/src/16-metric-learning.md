---
title: Metric learning and face recognition
summary: Classifiers need every class at training time; recognising people does not. Learn an embedding where the same face lands close and different faces far apart, train it with the ArcFace angular-margin loss written from scratch, make it robust to face masks with occlusion augmentation, and enrol a new person from a single photo with a threshold you can justify.
topics: embeddings · cosine similarity · ArcFace · open-set recognition · verification · ROC · thresholds · occlusion augmentation
script: 16_metric_learning.py
sources: a face-recognition technical assessment I did (ArcFace + ResNet-50 + MTCNN)
---

## Why classification is the wrong framing

A face-recognition system for a building has a list of employees that changes every week. A classifier with one output per employee would need retraining for every new hire — with a few photos each. And it has no way to say "this person is **not** in the list": softmax always picks someone.

**Metric learning** changes the question. Instead of "which of these $K$ people is this?", learn a function $f$ that maps a face to a vector — an **embedding** — such that photos of the same person are close and photos of different people are far apart. Then:

- **Verification** ("is this the same person?"): compare the similarity of two embeddings with a threshold.
- **Identification** ("who is this?"): find the nearest embedding in a **gallery** of enrolled people (k-NN, lesson 01) — and answer "unknown" if even the nearest is not similar enough.
- **Enrolment**: add a person by storing the embedding of one photo. No retraining.

The network never needs to have seen the people it recognises. That is the property we will measure: training uses 120 people of *Labeled Faces in the Wild*; every number below is computed on 38 other people.

```python
--8<-- "16_metric_learning.py:config"
```

```python
--8<-- "16_metric_learning.py:data"
```

```text
3217 images of 158 people: 2434 images of 120 people for training, 783 images of 38 unseen people for testing
```

(LFW has 530 photos of George W. Bush and 10 of many others; capping at 40 per person keeps one identity from dominating training and evaluation.)

## The model and the ArcFace loss

The embedder is lesson 14's pretrained ResNet-18 with its classifier replaced by a projection to 128 numbers, normalised to unit length — every face becomes a point on a sphere, and the similarity of two faces is the **cosine** of the angle between them (a dot product).

```python
--8<-- "16_metric_learning.py:model"
```

How do you train "same close, different far"? The **triplet loss** pulls an anchor towards a positive and pushes it from a negative, but needs careful mining of hard triplets. **ArcFace** (Deng et al., 2019) — the loss my original notebook used through `pytorch-metric-learning` — trains like a classifier over the training identities, with two twists:

1. The logits are **cosines** between the embedding and one learnt direction $\mathbf{W}_j$ per identity, scaled by $s$: only angles matter.
2. The true class must win by an **angular margin** $m$: its logit uses $\cos(\theta_y + m)$ instead of $\cos\theta_y$.

$$L = -\log\frac{e^{s\cos(\theta_y + m)}}{e^{s\cos(\theta_y + m)} + \sum_{j \ne y} e^{s\cos\theta_j}}$$

The margin forces each identity into a tight cone, well separated from the others. The class directions $\mathbf{W}$ are thrown away after training; what remains is an embedding space where *new* identities also form tight, separated clusters.

```python
--8<-- "16_metric_learning.py:arcface"
```

## Training with masks

Faces at a building entrance may wear masks, scarves, or be partly hidden. My notebook added a `RandomMaskOcclusion` augmentation: half of the time, black out the lower face:

```python
--8<-- "16_metric_learning.py:occlusion"
```

```python
--8<-- "16_metric_learning.py:train"
```

```text
  epoch  1: ArcFace loss 16.704
  ...
  epoch 10: ArcFace loss 1.189
fine-tuning took 4.7 min
```

## Evaluation on people never seen

```python
--8<-- "16_metric_learning.py:evaluate"
```

```text
ImageNet features (frozen)  clean: rank-1 0.534, AUC 0.753 | masked probes: rank-1 0.086, AUC 0.625
ArcFace fine-tuned          clean: rank-1 0.719, AUC 0.910 | masked probes: rank-1 0.688, AUC 0.909
```

- **Rank-1**: for each test photo, is the most similar *other* photo (among 783) the same person?
- **AUC**: over all 306,000 pairs, the probability that a same-person pair is more similar than a different-person pair (lesson 06's ROC, for verification).

!fig 16-roc | Verification ROC curves on the 38 unseen people. Left: clean photos. Right: a masked photo compared with clean photos — the realistic case of a gallery of ID photos and a masked visitor.

Generic ImageNet features already know something about faces (AUC 0.75), but they collapse when half the face is hidden (rank-1 from 53% to 9%): they mostly encode the mouth, chin and background. The ArcFace embedding, trained with occlusion augmentation, loses almost nothing with a mask (AUC 0.910 → 0.909). Augmentation teaches the network what to *ignore* — here, it learnt to identify people from the eyes, brows and upper face.

!fig 16-similarities | Cosine similarity of same-person pairs (blue) and different-person pairs (red), unseen people only. Fine-tuning pushes the two distributions apart; the dashed line is the threshold for a 1% false-accept rate.

!fig 16-retrieval | Masked queries of two unseen people and the four most similar clean photos in the test set, with their cosine similarity.

## Open-set identification: the threshold

The nearest neighbour is always *someone*. Deciding when it is "nobody we know" requires a **threshold**, and the ROC curve is where it comes from: pick the false-accept rate you can tolerate, read off the similarity.

```python
--8<-- "16_metric_learning.py:open-set"
```

```text
threshold for a 1% false accept rate: cosine similarity > 0.557 (true accept rate 0.352)
newcomer: Adrien Brody | stranger: Angelina Jolie
newcomer, another photo : best match Adrien Brody           similarity 0.750 -> Adrien Brody
newcomer, masked photo  : best match Adrien Brody           similarity 0.752 -> Adrien Brody
stranger (not enrolled) : best match Catherine Zeta-Jones   similarity 0.654 -> Catherine Zeta-Jones
```

Adrien Brody, never seen in training and enrolled with **one** photo, is recognised in another photo and with a mask. Angelina Jolie, not enrolled, is wrongly accepted as Catherine Zeta-Jones. Two reasons, both worth remembering:

1. **A per-pair error rate grows with the gallery.** The threshold gives a 1% false-accept rate *per comparison*; an identification query compares against all 21 enrolled people, so the chance that at least one stranger–person pair crosses the threshold is roughly $1 - 0.99^{21} \approx 19\%$ — more for look-alikes. Large galleries need much stricter thresholds.
2. **This model is weak by production standards.** At 1% false accepts it only accepts 35% of genuine pairs. Five minutes of training on 2,400 photos of 120 people is a demonstration; real systems train on millions of faces (below) and reach a true-accept rate above 99% at a false-accept rate of 1 in 10⁵.

!!! bug "A distance is not a score"
    In the notebook, a query photo of a person who was *not* in the database returned "Neighbor 1: Label = Landon_Donovan Score: 1.1109", and after enrolling him, "Score: 0.9899" for his own label. I read these as similarity scores. They were **distances**: the k-NN index behind `get_nearest_neighbors` returns Euclidean distances between the unit-length embeddings (related to the cosine by $d^2 = 2 - 2\cos$), so lower is better, and 1.11 means a cosine of only about 0.4 — a poor match. The `CosineSimilarity` I had passed to the match finder is not what the neighbour search used. The ranking was right, the interpretation backwards. And with no threshold, the first query confidently named a stranger. Two rules: know whether your number is a similarity or a distance, and choose an acceptance threshold on validation pairs before deploying.

!!! bug "Save the whole model"
    The same notebook saved only the trunk (`torch.save(trainer.models["trunk"].state_dict(), ...)`) and then built the inference model from the trunk *and the embedder still in memory*. It worked in that session; in a fresh one, the embedder would have been randomly initialised and every embedding meaningless. Save every module the inference path uses — here the whole `Embedder`, projection included — and test the reload (lesson 12).

## In practice

- Production systems first **detect and align** the face (my notebook used MTCNN): a crop centred on the eyes at a fixed scale removes most of the variation the embedder would otherwise have to learn. LFW images are roughly centred already.
- They are trained on millions of identities (MS1M, WebFace), reaching >99.8% verification accuracy on LFW; the principles are the ones above.
- Face recognition is biometric data, regulated in many countries (LGPD in Brazil, GDPR in Europe): consent, storage of embeddings and error rates across demographic groups are part of the engineering, not an afterthought.

## Exercises

1. Train without occlusion augmentation (`p_occlude=0`) and compare the masked-probe numbers.
2. Replace ArcFace by a plain softmax classifier over the 120 identities (keep the normalised embedding). How much worse is verification on unseen people? Then try the margin $m = 0.2$ and $0.6$.
3. Enrol each unseen person with the **mean** embedding of 3 photos instead of one. How does rank-1 identification change?
