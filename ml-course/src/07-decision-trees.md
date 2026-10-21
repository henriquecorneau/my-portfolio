---
title: Decision trees
summary: A model you can read like a flowchart. Measure impurity with the Gini index, grow a CART tree recursively from scratch and check it against scikit-learn, then watch depth drive overfitting and see why a single tree is so unstable — the reason the next lesson exists.
topics: Gini impurity · greedy splitting · recursion · depth · overfitting · variance
script: 07_decision_trees.py
sources: INSA IML lab on decision trees (TP1)
---

## Twenty questions

A decision tree classifies by asking a sequence of yes/no questions about the features — *is the petal width ≤ 0.8 cm?* — each answer leading to the next question, until a **leaf** gives the class. It needs no scaling (a threshold doesn't care about units), handles mixed feature types, and the result can be printed and checked by a domain expert. Those are big advantages in practice.

Learning a tree means choosing the questions. Finding the *optimal* tree is NP-hard, so every algorithm is **greedy**: at each node, pick the single question that best separates the classes *right now*, split the data, and recurse.

## Measuring purity

A node is good if it is **pure** — all its samples have the same class. The **Gini impurity** of a node with class proportions $p_k$ is

$$G = 1 - \sum_k p_k^2$$

the probability that two samples drawn at random from the node have different labels: 0 for a pure node, $1 - 1/K$ for a uniform mix of $K$ classes. (Entropy, $-\sum_k p_k\log p_k$, is the other common choice; they rarely give different trees.) A split is scored by the **weighted** impurity of its two children, and the best split is the one that lowers it most. Candidate thresholds are the midpoints between consecutive values of each feature.

```python
--8<-- "07_decision_trees.py:gini"
```

## CART from scratch

The tree is a recursive structure: a leaf, or a question with two subtrees. We stop when the node is pure, too small, or the maximum depth is reached.

```python
--8<-- "07_decision_trees.py:tree"
```

On the iris flowers (two features, petal length and width, three species):

```python
--8<-- "07_decision_trees.py:iris"
```

```text
root split: petal length (cm) <= 2.45, Gini impurity 0.667 -> 0.333
|--- petal width <= 0.80
|   |--- class: 0
|--- petal width >  0.80
|   |--- petal width <= 1.75
|   |   |--- petal length <= 4.95
|   |   |   |--- class: 1
|   |   |--- petal length >  4.95
|   |   |   |--- class: 2
|   |--- petal width >  1.75
...
depth-3 tree: training accuracy 0.973 (ours) vs 0.973 (scikit-learn), same predictions on 100%
```

The first question isolates *setosa* perfectly, from 0.667 to 0.333 impurity. Our tree asks it with petal length, scikit-learn with petal width: both separate setosa perfectly, it is a **tie**, and scikit-learn breaks ties by visiting the features in random order. The predictions are identical. Notice also the last split of scikit-learn's tree: both children predict class 2. It is not useless — it makes one child purer, which is what Gini rewards — but it does not change any prediction.

!fig 07-iris | Decision regions of trees of depth 1, 3 and unlimited. Trees cut the plane into axis-aligned rectangles; the unlimited tree carves a tiny box around a single flower.

## Depth and overfitting

A tree grown until every leaf is pure has **zero training error** — it can always isolate each sample in its own leaf. On a harder synthetic problem (2,000 samples, 20 informative features, overlapping classes):

```python
--8<-- "07_decision_trees.py:depth"
```

```text
depth with lowest held-out error: 11 (0.240); depth 20: train 0.000, held-out 0.252
```

!fig 07-depth | Training error falls to zero as the tree deepens; held-out error stops improving around depth 8–11 and stays at ~25%. The gap between the curves is overfitting.

!!! note "A gap is not the same as getting worse"
    In my lab report I wrote that trees deeper than 9 overfit and the best ones were depths 1 to 5. The first half is right — the train/held-out gap explodes — but the shallow trees are clearly *worse* here (28–41% error): they **underfit**. On this problem the held-out error does not rise again after its minimum; deep trees memorise the training set without (much) extra damage on new data. The best tree is still only 76% accurate, though, and the next figure shows why one tree cannot do better.

## The real problem: variance

Train the same unlimited tree on eight different samples of 100 points from the *same* distribution:

```python
--8<-- "07_decision_trees.py:instability"
```

!fig 07-instability | Fully grown trees on eight random samples of the two-moons distribution. Same algorithm, same distribution, wildly different boundaries.

A small change in the data changes the first split, which changes everything below it. In statistical terms, deep trees have **low bias** (they can represent almost anything) and **very high variance** (what they represent depends heavily on the particular sample). Pruning or limiting depth trades variance for bias. The much better idea — average many trees, so their individual quirks cancel out — is the subject of lesson 08.

## Exercises

1. Replace Gini by entropy in `best_split`. Does the iris tree change?
2. Add **regression trees**: a leaf predicts the mean of its samples, and a split minimises the weighted variance of the children. Fit $y = \sin(x)$ + noise and plot the staircase prediction for depths 2, 4 and 8.
3. `best_split` tries every threshold of every feature, $O(d\,N^2)$ per node. Sort each feature once and update the class counts incrementally while scanning thresholds to make it $O(d\,N\log N)$.
