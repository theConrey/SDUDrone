"""
Isolate an animal from a grass background using only:
  - color spaces (Lab)
  - histograms
  - Euclidean distance
  - Otsu thresholding

No hardcoded color range. Instead, we let the image tell us what "grass"
looks like by finding the most common color (the histogram peak).
"""

import cv2
import numpy as np
from icecream import ic


def find_dominant_color(lab_img, bins=32):
    """Build a 2D histogram of the a/b channels (color, without lightness)
    and return the color at the tallest bin. Since grass covers most of
    the photo, whatever color is most common IS the background color for
    this image -- no need to hardcode a hue range."""
    a = lab_img[:, :, 1]
    b = lab_img[:, :, 2]

    hist, a_edges, b_edges = np.histogram2d(
        a.flatten(), b.flatten(), bins=bins, range=[[0, 255], [0, 255]]
    )

    # Index of the tallest bin
    peak_idx = np.unravel_index(np.argmax(hist), hist.shape)
    a_peak = (a_edges[peak_idx[0]] + a_edges[peak_idx[0] + 1]) / 2
    b_peak = (b_edges[peak_idx[1]] + b_edges[peak_idx[1] + 1]) / 2

    return a_peak, b_peak


def build_distance_mask(img_bgr, blur_ksize=41):
    # Blur first -- individual blades of grass create pixel-to-pixel noise
    # that would otherwise show up as a false "far from background" signal.
    # Turned up from 15 -> 41 since the edited image has stronger local
    # contrast that would otherwise survive a lighter blur.
    blurred = cv2.GaussianBlur(img_bgr, (blur_ksize, blur_ksize), 0)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)

    a_peak, b_peak = find_dominant_color(lab)
    ic(a_peak, b_peak)

    a = lab[:, :, 1].astype(np.float32)
    b = lab[:, :, 2].astype(np.float32)

    # Euclidean distance from the dominant (background) color.
    # We skip the L channel on purpose -- lightness changes a lot across
    # the field from shadow/sun, but the actual color (a, b) is more
    # consistent for grass, even when the grass itself is patchy.
    dist = np.sqrt((a - a_peak) ** 2 + (b - b_peak) ** 2)
    dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Otsu automatically finds the best split point in this distance
    # image's histogram, instead of us picking a threshold by hand
    _, mask = cv2.threshold(dist_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return mask, dist_norm


def clean_mask(mask, open_ksize=5, close_ksize=25):
    """Two different kernel sizes doing two different jobs:

    - OPEN with a small kernel: erodes away tiny speckle noise (single
      stray grass blades that slipped through), then dilates back so real
      blobs return to roughly their original size.
    - CLOSE with a BIGGER kernel: dilates first to bridge small gaps
      between nearby blobs (like a dark mane splitting off from a lighter
      body), then erodes back down. The kernel needs to be at least as
      wide as the gap you're trying to bridge, which is why it's bigger
      than the opening kernel.
    """
    mask = cv2.medianBlur(mask, 7)

    open_kernel = np.ones((open_ksize, open_ksize), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)

    close_kernel = np.ones((close_ksize, close_ksize), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)

    return mask


def pick_animal_contour(mask, min_area=300, max_area_frac=0.03):
    """The distance+Otsu step can flag more than one region (e.g. a patch
    of differently lit grass, as well as the animal). Rather than just
    grabbing the largest blob -- which grabs whichever outlier happens to
    be biggest, even if it's a lighting patch -- we throw out anything
    implausibly small (noise) or implausibly large (can't be the animal,
    given how small it is relative to the frame in these photos), then
    take the largest of what's left."""
    img_area = mask.shape[0] * mask.shape[1]
    max_area = img_area * max_area_frac

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = [
        c for c in contours
        if min_area <= cv2.contourArea(c) <= max_area
    ]

    if not candidates:
        return None

    return max(candidates, key=cv2.contourArea)


def main(path_for_mask="./input/img_24_edited.jpg", path_original="./input/img_24.jpg", out_dir="./output/flight"):
    mask_source = cv2.imread(path_for_mask)
    if mask_source is None:
        raise FileNotFoundError(path_for_mask)

    original = cv2.imread(path_original)
    if original is None:
        raise FileNotFoundError(path_original)

    if mask_source.shape[:2] != original.shape[:2]:
        raise ValueError(
            f"Edited image {mask_source.shape[:2]} and original "
            f"{original.shape[:2]} are different sizes -- GIMP export must "
            f"match the original resolution exactly for the mask to line up."
        )

    raw_mask, dist_debug = build_distance_mask(mask_source)
    cv2.imwrite(f"{out_dir}/debug_distance.png", dist_debug)
    cv2.imwrite(f"{out_dir}/debug_raw_mask.png", raw_mask)

    mask = clean_mask(raw_mask)
    cv2.imwrite(f"{out_dir}/debug_clean_mask.png", mask)

    animal_contour = pick_animal_contour(mask)

    final_mask = np.zeros_like(mask)
    if animal_contour is not None:
        cv2.drawContours(final_mask, [animal_contour], -1, 255, thickness=cv2.FILLED)
    else:
        ic("No plausible animal-sized region found -- check debug images / thresholds")

    cv2.imwrite(f"{out_dir}/final_mask.png", final_mask)

    # Apply the mask (derived from the edited image) onto the ORIGINAL image
    isolated = cv2.bitwise_and(original, original, mask=final_mask)
    cv2.imwrite(f"{out_dir}/final_isolated.png", isolated)

    # Draw the bounding box on the ORIGINAL image too
    boxed = original.copy()
    if animal_contour is not None:
        x, y, w, h = cv2.boundingRect(animal_contour)
        cv2.rectangle(boxed, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imwrite(f"{out_dir}/final_bound_box.png", boxed)


if __name__ == "__main__":
    import os
    os.makedirs("./output/flight", exist_ok=True)
    main()
