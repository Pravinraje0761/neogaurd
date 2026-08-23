"""
NeoGuard - Uncalibrated Illumination Normalization (real-data adaptation)
-------------------------------------------------------------------------------
color_calibration.py corrects each capture against a known reference target
physically built into the NeoGuard attachment. None of the real public datasets
pulled into this project include such a target -- they are ordinary clinical or
macro photos with arbitrary, unknown lighting. There is no way to recover true
calibration without a reference in the frame.

As a partial, honest substitute, this module applies a standard Gray-World
white-balance assumption (the average scene reflectance is neutral gray) to
reduce gross illumination-color bias before feature extraction. This is
materially weaker than true calibration -- it can't correct exposure, and it can
be wrong when a scene genuinely isn't gray-averaged (e.g. a very yellow-dominant
incubator interior) -- but it is standard practice for uncalibrated-camera color
analysis and is better than analyzing raw uncorrected pixels.

This gap -- no calibration target in any currently-available real dataset -- is
one of the concrete reasons real accuracy on this data should be read as a lower
bound / sanity check, not a final number. See the technical report's real-data
section.
"""

import cv2
import numpy as np


def gray_world_normalize(bgr_image: np.ndarray) -> np.ndarray:
    img = bgr_image.astype(np.float32)
    means = img.reshape(-1, 3).mean(axis=0)  # B, G, R
    overall_mean = means.mean()
    gains = overall_mean / np.clip(means, 1e-3, None)
    # Clip gains to a sane range so near-black/near-white scenes don't blow up
    gains = np.clip(gains, 0.5, 2.0)
    normalized = img * gains
    return np.clip(normalized, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    demo = np.full((50, 50, 3), (100, 150, 200), dtype=np.uint8)  # warm cast
    out = gray_world_normalize(demo)
    print("before mean (BGR):", demo.reshape(-1, 3).mean(axis=0))
    print("after mean (BGR):", out.reshape(-1, 3).mean(axis=0))
