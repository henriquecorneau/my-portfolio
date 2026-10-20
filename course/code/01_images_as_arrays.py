"""Lesson 01 - Images are arrays: NumPy, dtypes, masks and OpenCV basics."""
import cv2
import numpy as np

import cvutils as U

U.setup("01")

# --8<-- [start:dtype]
a = np.full((2, 2), 255, dtype=np.uint8)
b = np.full((2, 2), 255, dtype=np.uint16)
print("uint8  255 + 1 ->", (a + 1)[0, 0])          # wraps around to 0
print("uint16 255 + 1 ->", (b + 1)[0, 0])          # 256, fits in 16 bits
print("cv2.add saturates ->", cv2.add(a, 1)[0, 0])  # stays at 255
print("safe NumPy sum ->", (a.astype(np.int32) + 1)[0, 0])
# --8<-- [end:dtype]

# --8<-- [start:masks]
A = np.random.randint(0, 10, (5, 5))
mask = A < 5                 # boolean matrix, same shape as A
print(A[mask])               # 1-D array with the selected values

B = A.copy()
B[A < 5] = 0                 # write through a mask
B[(A % 2).astype(bool)] -= 1  # odd values minus one
in_range = np.logical_and(A > 4, A < 7)
print(B, in_range, sep="\n")
# --8<-- [end:masks]

# --8<-- [start:read]
gray = U.imread("basics/cameraman.tif", cv2.IMREAD_GRAYSCALE)  # flag 0
color = U.imread("basics/rouen.jpg")                            # flag 1, BGR!

print("gray :", gray.shape, gray.dtype)    # (256, 256) uint8
print("color:", color.shape, color.dtype)  # (576, 800, 3) uint8
print("pixel (50, 100):", gray[50, 100])   # row 50, column 100
b, g, r = color[0, 0]                      # first pixel, in B, G, R order
# --8<-- [end:read]

# --8<-- [start:bgr]
wrong = color[:, :, ::-1]   # what Matplotlib "sees" if we forget the conversion
rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
# --8<-- [end:bgr]
U.show("bgr", [rgb[:, :, ::-1], wrong], ["cvtColor(BGR2RGB)", "BGR shown as RGB"], size=4.2)

# --8<-- [start:channels]
blue, green, red = cv2.split(color)
only_red = color.copy()
only_red[:, :, 0:2] = 0     # zero the B and G planes
# --8<-- [end:channels]
U.show("channels", [blue, green, red, only_red],
       ["B channel", "G channel", "R channel", "red plane only"], cols=4, size=3.2)

# --8<-- [start:gray]
f = color.astype(np.float32)
gray_mean = (f[:, :, 0] + f[:, :, 1] + f[:, :, 2]) / 3
# ITU-R BT.601 luma: the eye is most sensitive to green, least to blue
gray_luma = 0.114 * f[:, :, 0] + 0.587 * f[:, :, 1] + 0.299 * f[:, :, 2]
gray_cv = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
print("max |luma - cv2|:", np.abs(gray_luma - gray_cv).max())  # < 1, just rounding
# --8<-- [end:gray]
U.show("gray", [gray_mean.astype(np.uint8), gray_cv],
       ["(B + G + R) / 3", "cvtColor(BGR2GRAY)"], size=4.2)

# --8<-- [start:roi]
h, w = gray.shape
quarter = gray[h // 2:, : w // 2]     # bottom-left quarter: a *view*, no copy
quarter_copy = gray[h // 2:, : w // 2].copy()

canvas = color.copy()
canvas[250:300, 380:430] = (0, 0, 255)  # paint a red square (BGR)
cv2.rectangle(canvas, (100, 80), (300, 220), (255, 0, 0), 4)
cv2.circle(canvas, (600, 150), 60, (0, 255, 0), -1)
cv2.putText(canvas, "Rouen", (40, 540), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
# --8<-- [end:roi]
U.show("roi", [quarter, canvas], ["gray[h//2:, :w//2]", "drawing on a copy"], size=4.2)

# --8<-- [start:blend]
logo = U.imread("basics/logo_color.png")
logo = cv2.resize(logo, (color.shape[1], color.shape[0]))
alpha = 0.5
blend_np = (alpha * color + (1 - alpha) * logo).astype(np.uint8)  # float math, then cast
blend_cv = cv2.addWeighted(color, alpha, logo, 1 - alpha, 0)
# --8<-- [end:blend]

# --8<-- [start:synthetic]
ramp = np.tile(np.arange(256, dtype=np.uint8), (256, 1))  # M[i, j] = j
# --8<-- [end:synthetic]
U.show("synthetic", [ramp, blend_cv], ["M[i, j] = j", "addWeighted(0.5)"], size=4.2)

# --8<-- [start:transpose]
transposed = gray.T            # rows become columns
rot90 = np.rot90(gray)         # counter-clockwise rotation
flipped = cv2.flip(gray, 1)    # mirror around the vertical axis
# --8<-- [end:transpose]
U.show("transpose", [gray, transposed, rot90, flipped],
       ["original", "gray.T", "np.rot90", "cv2.flip(…, 1)"], cols=4, size=3)
