import cv2
import numpy as np

def find_peak_px(img_path, window=5):
    # read & grayscale
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # column‐sum intensity
    profile = np.sum(gray, axis=0).astype(float)
    # smooth with a simple moving average
    kern = np.ones(window)/window
    smooth = np.convolve(profile, kern, mode='same')
    # peak at the max intensity
    px = int(np.argmax(smooth))
    return px

# Example:
p1 = find_peak_px("c-fitc-calib.png")
p2 = find_peak_px("c-tritc-calib.png")
λ1, λ2 = 501.0, 557.0
print("FITC peak px:", p1, "TRITC peak px:", p2)


# slope (nm per pixel):
a = (λ2 - λ1) / (p2 - p1)

# intercept (wavelength at pixel=0):
b = λ1 - a * p1

print(f"a = {a:.4f}  nm/px")
print(f"b = {b:.2f}   nm")

