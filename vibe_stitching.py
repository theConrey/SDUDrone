"""
Stitch a folder of overlapping images into a single panorama.

Usage:
    python stitch.py --folder inputs --output panorama.jpg --mode panorama
    python stitch.py --folder drone_shots --output aerial.jpg --mode scans
"""

import argparse
import sys
from pathlib import Path

import cv2


def load_images(folder="inputs"):
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
    """
    Stitch a list of images together.

    mode="panorama" -> cv2.Stitcher_PANORAMA: assumes a shared viewpoint
                        (handheld pan, images rotate around a point). Applies
                        lens/perspective correction.
    mode="scans"     -> cv2.Stitcher_SCANS: better for top-down drone/aerial
                         imagery where the camera translates over a roughly
                         planar scene rather than rotating.
    """
    if len(images) < 2:
        raise ValueError("Need at least 2 images to stitch")

    cv_mode = cv2.Stitcher_PANORAMA if mode == "panorama" else cv2.Stitcher_SCANS
    stitcher = cv2.Stitcher_create(cv_mode)

    status, pano = stitcher.stitch(images)

    status_messages = {
        cv2.Stitcher_ERR_NEED_MORE_IMGS: "Not enough images with matching features between them",
        cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "Homography estimation failed - images may not overlap enough or lack distinct features",
        cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "Camera parameter adjustment failed",
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


def main():
    parser = argparse.ArgumentParser(description="Stitch images into a panorama")
    parser.add_argument("--folder", default="inputs", help="Folder containing input images")
    parser.add_argument("--output", default="panorama.jpg", help="Output file path")
    parser.add_argument(
        "--mode", choices=["panorama", "scans"], default="panorama",
        help="'panorama' for handheld/rotating shots, 'scans' for top-down drone/aerial shots"
    )
    parser.add_argument("--no-crop", action="store_true", help="Skip cropping the black border")
    args = parser.parse_args()

    print(f"Loading images from '{args.folder}'...")
    images = load_images(args.folder)
    print(f"Loaded {len(images)} image(s)")

    if len(images) < 2:
        print("Need at least 2 valid images to stitch. Exiting.")
        sys.exit(1)

    print(f"Stitching in '{args.mode}' mode...")
    pano = stitch_images(images, mode=args.mode)

    if not args.no_crop:
        pano = crop_black_border(pano)

    cv2.imwrite(args.output, pano)
    print(f"Saved panorama to '{args.output}' ({pano.shape[1]}x{pano.shape[0]})")


if __name__ == "__main__":
    main()