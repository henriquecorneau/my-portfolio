"""Lesson 05 - Noise and spatial filtering: mean, median, Gaussian, convolution."""
import time

import cv2
import numpy as np

import cvutils as U

U.setup("05")

clean = U.imread("basics/cameraman.tif", cv2.IMREAD_GRAYSCALE)
noisy_g = U.imread("noise/noisy_cameraman_G.tif", cv2.IMREAD_GRAYSCALE)
noisy_sp = U.imread("noise/noisy_cameraman_SP.tif", cv2.IMREAD_GRAYSCALE)


# --8<-- [start:noise]
def add_gaussian_noise(img, sigma):
    noise = np.random.normal(0, sigma, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def add_salt_pepper(img, p):
    """Each pixel is hit with probability p; half of the hits are black, half white."""
    out = img.copy()
    r = np.random.random(img.shape)
    out[r < p / 2] = 0
    out[(r >= p / 2) & (r < p)] = 255
    return out


def psnr(ref, img):
    """Peak signal-to-noise ratio in dB (higher is better)."""
    mse = np.mean((ref.astype(np.float64) - img) ** 2)
    return 10 * np.log10(255 ** 2 / mse)
# --8<-- [end:noise]

# --8<-- [start:diff]
diff_g = noisy_g.astype(np.int16) - clean       # signed! uint8 would wrap around
diff_sp = noisy_sp.astype(np.int16) - clean
print("Gaussian noise: mean %.2f std %.2f" % (diff_g.mean(), diff_g.std()))
print("S&P: %.1f%% of the pixels changed" % (100 * np.mean(diff_sp != 0)))
# --8<-- [end:diff]

fig, axes = U.figure(4, 1, size=(3.2, 3.0))
axes[0, 0].imshow(noisy_g, cmap="gray", vmin=0, vmax=255)
axes[0, 0].set_title("Gaussian noise")
axes[0, 1].imshow(noisy_sp, cmap="gray", vmin=0, vmax=255)
axes[0, 1].set_title("salt & pepper noise")
axes[0, 2].hist(diff_g.ravel(), bins=61, color="#2563eb")
axes[0, 2].set_title("noisy_G − clean")
axes[0, 3].hist(diff_sp.ravel(), bins=61, color="#dc2626", log=True)
axes[0, 3].set_title("noisy_SP − clean (log)")
for ax in axes[0, :2]:
    ax.axis("off")
U.finish("noise", fig)

# --8<-- [start:averaging]
frames = [U.imread(f"frames/galaxia{i}.jpg") for i in range(9)]
mean = np.zeros(frames[0].shape, np.float64)
for i, frame in enumerate(frames):
    alpha = 1 / (i + 1)                          # running (incremental) mean
    mean = (1 - alpha) * mean + alpha * frame
stacked = mean.astype(np.uint8)
# --8<-- [end:averaging]
U.show("averaging", [frames[0], stacked], ["one exposure", "mean of 9 exposures"], size=4.4)


# --8<-- [start:mean_median]
def mean_filter(img, n):
    """n x n box filter, borders left at zero."""
    k = n // 2
    out = np.zeros_like(img)
    for i in range(k, img.shape[0] - k):
        for j in range(k, img.shape[1] - k):
            out[i, j] = img[i - k:i + k + 1, j - k:j + k + 1].mean()  # note the +1
    return out


def median_filter(img, n):
    k = n // 2
    out = np.zeros_like(img)
    for i in range(k, img.shape[0] - k):
        for j in range(k, img.shape[1] - k):
            out[i, j] = np.median(img[i - k:i + k + 1, j - k:j + k + 1])
    return out
# --8<-- [end:mean_median]


# --8<-- [start:convolution]
def convolve(img, kernel):
    """2-D correlation of img with kernel (equal to convolution for symmetric kernels)."""
    kh, kw = kernel.shape[0] // 2, kernel.shape[1] // 2
    src = img.astype(np.float32)
    out = np.zeros_like(src)
    for i in range(kh, img.shape[0] - kh):
        for j in range(kw, img.shape[1] - kw):
            window = src[i - kh:i + kh + 1, j - kw:j + kw + 1]
            out[i, j] = np.sum(window * kernel)
    return out


def gaussian_kernel(size, sigma):
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return k / k.sum()                           # weights sum to 1: brightness preserved


box3 = np.ones((3, 3)) / 9
gauss3 = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16            # sigma ~ 0.8
expo5 = np.array([[1, 1, 3, 1, 1], [1, 3, 7, 3, 1], [3, 7, 16, 7, 3],
                  [1, 3, 7, 3, 1], [1, 1, 3, 1, 1]]) / 80             # exponential filter
# --8<-- [end:convolution]

results = {}
for name, img in [("G", noisy_g), ("SP", noisy_sp)]:
    results[name] = {
        "mean 5x5": mean_filter(img, 5),
        "median 3x3": median_filter(img, 3),
        "Gaussian 5x5": np.clip(convolve(img, gaussian_kernel(5, 1.0)), 0, 255).astype(np.uint8),
        "exponential 5x5": np.clip(convolve(img, expo5), 0, 255).astype(np.uint8),
    }
    for k, v in results[name].items():
        b = 2  # ignore the zero border when scoring
        print(f"{name:>2} {k:<16} PSNR {psnr(clean[b:-b, b:-b], v[b:-b, b:-b]):5.1f} dB "
              f"(noisy: {psnr(clean, img):.1f} dB)")

for name, img in [("G", noisy_g), ("SP", noisy_sp)]:
    r = results[name]
    U.show(f"filters-{name.lower()}", [img] + list(r.values()), ["noisy"] + list(r.keys()), cols=5, size=2.8)

# --8<-- [start:opencv]
box = cv2.blur(noisy_g, (5, 5))
gauss = cv2.GaussianBlur(noisy_g, (5, 5), sigmaX=1.0)
median = cv2.medianBlur(noisy_sp, 3)
custom = cv2.filter2D(noisy_g, ddepth=-1, kernel=expo5.astype(np.float32))  # any kernel
bilateral = cv2.bilateralFilter(noisy_g, d=7, sigmaColor=40, sigmaSpace=5)  # edge-preserving
# --8<-- [end:opencv]

mine = results["G"]["exponential 5x5"]
inner = (slice(2, -2), slice(2, -2))
print("filter2D vs ours (inside the border): max diff =", np.abs(custom[inner].astype(int) - mine[inner]).max())

t0 = time.perf_counter()
convolve(noisy_g, expo5)
t1 = time.perf_counter()
cv2.filter2D(noisy_g, -1, expo5)
t2 = time.perf_counter()
print(f"python loop {1000 * (t1 - t0):.0f} ms vs filter2D {1000 * (t2 - t1):.2f} ms")

U.show("median-size", [noisy_sp, cv2.medianBlur(noisy_sp, 3), cv2.medianBlur(noisy_sp, 7),
                       cv2.medianBlur(noisy_sp, 15)],
       ["noisy", "median 3", "median 7", "median 15"], cols=4, size=3)

# --8<-- [start:barcode]
barcode = U.imread("noise/barcode.png", cv2.IMREAD_GRAYSCALE)
smooth = cv2.GaussianBlur(barcode, (5, 5), 0)                    # 1. remove the grain
background = cv2.blur(smooth, (101, 101))                        # 2. a huge mean keeps only the lighting
flat = cv2.subtract(background, smooth)                          # 3. bars become bright, gradient gone
flat = cv2.normalize(flat, None, 0, 255, cv2.NORM_MINMAX)        # 4. use the full range
_, bars = cv2.threshold(flat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 5. automatic threshold
result = 255 - bars                                              # black bars on white
naive = cv2.threshold(barcode, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
# --8<-- [end:barcode]
U.show("barcode", [barcode, naive, background, flat, result],
       ["input", "Otsu on the input", "background (101x101 mean)", "background − image", "final"],
       cols=5, size=3)
