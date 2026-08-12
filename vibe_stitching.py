"""
Stitch a folder of overlapping images into a single panorama.

Just edit the settings below, then run this script directly
(e.g. hit "Run" in your editor, or `python stitch.py` with no arguments).
"""

from pathlib import Path
import cv2

# ─── Settings — edit these ────────────────────────────────────────────
INPUT_FOLDER = "Test_Images"        # folder containing your images
OUTPUT_FILE = "lion.jpg"   # name/path of the stitched result
MODE = "scans"              # "panorama" = handheld/rotating shots
                                # "scans"    = top-down drone/aerial shots
CROP_BORDER = True              # trim the black edges left after stitching
# ───────────────────────────────────────────────────────────────────────


def load_images(folder):
    """Load every valid image file from a folder, sorted by filename."""
    valid_ext = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp"}
    images = []
    for path in sorted(Path(folder).iterdir()):
        if path.suffix.lower() in valid_ext:
            img = cv2.imread(str(path))
            if img is not None:
                images.append(img)
            else:
                print(f"Warning: failed to load {path}")
    return images


def stitch_images(images, mode="panorama"):
    """Stitch a list of images into one panorama using OpenCV's Stitcher."""
    if len(images) < 2:
        raise ValueError("Need at least 2 images to stitch")

    cv_mode = cv2.Stitcher_PANORAMA if mode == "panorama" else cv2.Stitcher_SCANS
    stitcher = cv2.Stitcher_create(cv_mode)
    status, pano = stitcher.stitch(images)

    status_messages = {
        cv2.Stitcher_ERR_NEED_MORE_IMGS:
            "Not enough images with matching features between them",
        cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL:
            "Homography estimation failed - images may not overlap enough or lack distinct features",
        cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL:
            "Camera parameter adjustment failed",
    }
    if status != cv2.Stitcher_OK:
        msg = status_messages.get(status, f"Unknown error (status={status})")
        raise RuntimeError(f"Stitching failed: {msg}")

    return pano


def crop_black_border(pano):
    """Trim the black/empty border the stitcher usually leaves behind."""
    gray = cv2.cvtColor(pano, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return pano
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return pano[y:y + h, x:x + w]


# ─── Run ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Loading images from '{INPUT_FOLDER}'...")
    images = load_images(INPUT_FOLDER)
    print(f"Loaded {len(images)} image(s)")

    if len(images) < 2:
        raise SystemExit("Need at least 2 valid images to stitch.")

    print(f"Stitching in '{MODE}' mode...")
    pano = stitch_images(images, mode=MODE)

    if CROP_BORDER:
        pano = crop_black_border(pano)

    cv2.imwrite(OUTPUT_FILE, pano)
    print(f"Saved panorama to '{OUTPUT_FILE}' ({pano.shape[1]}x{pano.shape[0]})")