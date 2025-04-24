import cv2
import numpy as np

def get_intensity_profile(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    prof = np.sum(img, axis=0).astype(float)
    return prof

def refine_peak_position(profile, idx):
    # take three points around idx: idx-1, idx, idx+1
    if idx <= 0 or idx >= len(profile)-1:
        return float(idx)
    x = np.array([idx-1, idx, idx+1], dtype=float)
    y = profile[x.astype(int)]
    # fit y = ax^2 + bx + c
    coeff = np.polyfit(x, y, 2)
    a, b, _ = coeff
    if a == 0:
        return float(idx)
    # vertex at x = -b/(2a)
    return -b / (2*a)

# load and smooth profiles
win = 5
kern = np.ones(win)/win

prof1 = np.convolve(get_intensity_profile("c-fitc-calib.png"), kern, mode='same')
prof2 = np.convolve(get_intensity_profile("c-tritc-calib.png"), kern, mode='same')

# rough peaks
rough1 = int(np.argmax(prof1))
rough2 = int(np.argmax(prof2))

# refined peak positions (subpixel)
p1 = refine_peak_position(prof1, rough1)
p2 = refine_peak_position(prof2, rough2)

λ1, λ2 = 501.0, 557.0  # known LED peaks

# slope and intercept
a = (λ2 - λ1) / (p2 - p1)
b = λ1 - a * p1

print(f"Rough FITC px: {rough1}, refined: {p1:.3f}")
print(f"Rough TRITC px: {rough2}, refined: {p2:.3f}")
print(f"\nCalibration constants:")
print(f"  a = {a:.6f}  nm/px")
print(f"  b = {b:.6f}  nm")

# Verify by mapping the FITC peak back to wavelength
estimated_λ1 = a * p1 + b
print(f"\nEstimated FITC λ using new constants: {estimated_λ1:.3f} nm (should ≈ {λ1} nm)")
