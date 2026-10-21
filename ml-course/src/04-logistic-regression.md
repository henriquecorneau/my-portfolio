---
title: Logistic regression
summary: The workhorse of classification. Derive the cross-entropy loss from probabilities, train logistic regression by gradient descent in ten lines, build a spam filter whose weights you can read, and extend it to ten classes with the softmax — the exact output layer of every classification network in Part IV.
topics: sigmoid · cross-entropy · maximum likelihood · log transform · odds · softmax · MNIST
script: 04_logistic_regression.py
sources: INSA IML lab 2 (logistic regression), UFRN Deep Learning class 7
---

## From a score to a probability

For classification we want $P(y = 1 \mid \mathbf{x})$, a number between 0 and 1. Logistic regression computes a linear **score** $z = \mathbf{w}^\top\mathbf{x} + b$, exactly as in lesson 02, and squashes it with the **sigmoid**:

$$P(y=1 \mid \mathbf{x}) = \sigma(z) = \frac{1}{1 + e^{-z}}$$

The decision rule "predict 1 if $P > 0.5$" is the same as "$z > 0$": the **decision boundary is the hyperplane** $\mathbf{w}^\top\mathbf{x} + b = 0$. Logistic regression is a *linear* classifier; the sigmoid only turns distances to the boundary into confidences.

Another way to read it: $z = \log\frac{P}{1-P}$, the **log-odds**. Each weight $w_j$ says how much the log-odds change when $x_j$ grows by one unit — that will make the spam filter readable.

## The loss: cross-entropy

Squared error is a bad fit for probabilities (with a sigmoid it is not convex, and it barely punishes confident mistakes). Take the same road as lesson 02 — **maximum likelihood**. Each label is a coin flip with probability $p_i = \sigma(z_i)$, so the likelihood of the data is $\prod_i p_i^{y_i}(1-p_i)^{1-y_i}$. Its negative logarithm, averaged, is the **binary cross-entropy**:

$$J(\mathbf{w}, b) = -\frac{1}{N}\sum_i \Big[y_i \log p_i + (1 - y_i)\log(1 - p_i)\Big]$$

A confident wrong answer ($p_i \to 0$ while $y_i = 1$) costs $-\log p_i \to \infty$. And the gradient is strikingly simple — thanks to $\sigma'(z) = \sigma(z)(1-\sigma(z))$, everything cancels to

$$\nabla_\mathbf{w} J = \frac{1}{N}\sum_i (p_i - y_i)\,\mathbf{x}_i$$

"error times input", the same form as linear regression. $J$ is convex: gradient descent finds the global minimum.

```python
--8<-- "04_logistic_regression.py:data"
```

```python
--8<-- "04_logistic_regression.py:scratch"
```

```text
from scratch: w = [0.8451 3.2715], b = 1.6196, loss 0.0641
scikit-learn: w = [0.8631 3.3297], b = 1.6758
training accuracy 0.980
```

Our weights are close to scikit-learn's but not identical: the two clouds barely overlap, so the loss keeps decreasing very slowly as the weights grow (on perfectly separable data the unregularised optimum is at infinity!). scikit-learn's L-BFGS solver, a quasi-Newton method (lesson 03), gets further than 5,000 plain gradient steps. In practice a small L2 penalty — scikit-learn's default, $C=1$ — keeps the weights finite.

!fig 04-synthetic | The sigmoid; the predicted probability of class 1 over the plane (the black line is P = 0.5, a straight line); and the cross-entropy during training.

## A spam filter

The *Spambase* dataset: 4,601 e-mails from Hewlett-Packard in the late 90s, each described by 57 numbers — the frequency (%) of 48 words (`free`, `remove`, `george`…), of 6 characters (`!`, `$`…), and three statistics of runs of capital letters.

```python
--8<-- "04_logistic_regression.py:spam"
```

```text
4601 e-mails: 2788 ham, 1813 spam; feature max values from 2.17 to 15841
```

```python
--8<-- "04_logistic_regression.py:spam-models"
```

```text
raw           train acc 0.9309  test acc 0.9302
standardised  train acc 0.9289  test acc 0.9257
log + std     train acc 0.9416  test acc 0.9433
```

The log transform is the big win. Word frequencies are **heavy-tailed**: most e-mails have 0, a few have a lot. On a linear scale the model can only say "more `remove` = more spam" proportionally, and a handful of extreme e-mails dominate the fit. After $\log(x + 0.01)$ the feature separates *absent* (a spike at $\log 0.01$) from *present* and compresses the tail, so "the word appears at all" — which is what matters — becomes easy to express.

!fig 04-spam-hist | The frequency of "remove" in ham and spam, raw and after the log transform (log counts on the y axis).

### Reading the weights

```python
--8<-- "04_logistic_regression.py:coefs"
```

```text
word_freq_george             -1.98   (odds x0.14 per std)
word_freq_hp                 -1.55   (odds x0.21 per std)
capital_run_length_total     +0.90   (odds x2.46 per std)
word_freq_meeting            -0.84   (odds x0.43 per std)
word_freq_cs                 -0.83   (odds x0.44 per std)
word_freq_remove             +0.79   (odds x2.20 per std)
...
```

The strongest *ham* signals are `george` and `hp`: the ham e-mails were donated by George Forman's colleagues at HP. A filter trained on this data would be useless for anyone else — the model learnt the dataset's quirks, a lesson about **where the data comes from** that no test accuracy will reveal. The spam side is what you'd expect: lots of capitals, `remove` ("to be removed from this list…"), `!` and `$`.

!fig 04-spam-coefs | The 15 largest weights of the log + standardised model. Red pushes towards spam, blue towards ham.

!!! warning "Compare weights only on a common scale"
    In my lab report I ranked features by the raw coefficients of the model trained on **unscaled** data. A weight is "change in log-odds per *unit* of the feature" — and units differ: one more percent of `george` is enormous, one more character in the total capital run length is nothing. Only after standardisation are weights "per standard deviation" and comparable. Even then, correlated features share their weight, so a small weight does not prove a feature is useless.

## Ten classes: the softmax

For $K$ classes, give each class its own score $z_k = \mathbf{w}_k^\top\mathbf{x} + b_k$ and turn the scores into probabilities with the **softmax**:

$$P(y = k \mid \mathbf{x}) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

All positive, summing to 1, and the largest score gets the largest probability. The loss is again the negative log-probability of the true class (**categorical cross-entropy**), and the gradient again "probability minus one-hot label, times input". This *multinomial logistic regression* is literally the last layer of every classification network: a CNN (lesson 13) is a learned feature extractor followed by softmax regression.

```python
--8<-- "04_logistic_regression.py:softmax"
```

```text
softmax regression on 10000 MNIST digits: train acc 0.9355, test acc 0.9104 (8 s)
weights: (10, 784)
```

!fig 04-softmax-weights | The 784 weights of each class, shown as 28×28 images (red positive, blue negative). Each is a blurry template: the "0" detector rewards ink on a ring and punishes ink in the centre.

Since the score is a dot product between the image and $\mathbf{w}_k$, each class weight vector *is* an image — a template the digit is matched against. 91% with templates is respectable, but one template per class cannot cope with all the ways of writing a 7. Neural networks will learn many intermediate templates (features) and combine them; that takes us from 91% to 99% in lesson 13.

## Exercises

1. Change the decision threshold from 0.5 to 0.9 on the spam filter. How many spams get through, and how many legitimate e-mails are blocked? Which threshold would you choose for a real mailbox? (Lesson 06 formalises this with precision and recall.)
2. Implement the softmax regression from scratch with NumPy (one weight matrix `W` of shape `(784, 10)`) and train it with mini-batch gradient descent. Subtract `z.max(axis=1, keepdims=True)` before the exponential — why?
3. Train the spam model without `george` and `hp`. How much accuracy is lost?
