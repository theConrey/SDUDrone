import cv2
import numpy as np

input_images = []

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


# def merge_two(img1:np.ndarray, img2:np.ndarray):
#     print("called merge")

#     # Initialize SIFT detector
#     sift = cv2.SIFT_create()

#     # Detect keypoints and descriptors
#     kp1, des1 = sift.detectAndCompute(img1, None)
#     kp2, des2 = sift.detectAndCompute(img2, None)

#     bf = cv2.BFMatcher(cv2.NORM_L2)
#     knn_matches = bf.knnMatch(des1, des2, k=2)

#     # Lowe's ratio test
#     good_matches = []
#     for m, n in knn_matches:
#         if m.distance < 0.75 * n.distance:
#             good_matches.append(m)

#     print(f"{len(good_matches)} good matches out of {len(knn_matches)}")

#     src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
#     dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
#     # Compute homography
#     # H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
#     H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

#     # Get the dimensions of the images
#     h1, w1 = img1.shape[:2]
#     h2, w2 = img2.shape[:2]

#     # Get the canvas dimesions
#     pts = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
#     dst = cv2.perspectiveTransform(pts, H)
#     img2_warped = cv2.warpPerspective(img2, H, (w1 + w2, h1))

#     # Place the first image on the canvas
#     img2_warped[0:h1, 0:w1] = img1
#     # Simple blending technique
#     result = img2_warped

#     return result

# def merge_two_vibe(img1: np.ndarray, img2: np.ndarray):
#     print("called merge vibe")

#     sift = cv2.SIFT_create()
#     kp1, des1 = sift.detectAndCompute(img1, None)
#     kp2, des2 = sift.detectAndCompute(img2, None)

#     # bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
#     # matches = bf.match(des1, des2)
#     # matches = sorted(matches, key=lambda x: x.distance)
#     bf = cv2.BFMatcher(cv2.NORM_L2)
#     knn_matches = bf.knnMatch(des1, des2, k=2)

#     # Lowe's ratio test
#     good_matches = []
#     for m, n in knn_matches:
#         if m.distance < 0.75 * n.distance:
#             good_matches.append(m)

#     print(f"{len(good_matches)} good matches out of {len(knn_matches)}")

#     src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
#     dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

#     # H maps img2 -> img1's coordinate frame
#     H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

#     h1, w1 = img1.shape[:2]
#     h2, w2 = img2.shape[:2]

#     # Corners of both images in img1's frame
#     corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
#     corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
#     warped_corners2 = cv2.perspectiveTransform(corners2, H)

#     all_corners = np.concatenate((corners1, warped_corners2), axis=0)

#     # Bounding box that fits everything
#     x_min, y_min = np.floor(all_corners.min(axis=0).ravel()).astype(int)
#     x_max, y_max = np.ceil(all_corners.max(axis=0).ravel()).astype(int)

#     # Translation so nothing lands at negative coordinates
#     translation = np.array([
#         [1, 0, -x_min],
#         [0, 1, -y_min],
#         [0, 0, 1]
#     ], dtype=np.float64)

#     canvas_w = x_max - x_min
#     canvas_h = y_max - y_min

#     result = cv2.warpPerspective(img2, translation @ H, (canvas_w, canvas_h))

#     # Place img1 at its translated offset
#     x_off, y_off = -x_min, -y_min
#     result[y_off:y_off + h1, x_off:x_off + w1] = img1

#     # Autocrop leftover black padding around the actual content
#     gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
#     nonzero_pts = cv2.findNonZero(gray)
#     if nonzero_pts is not None:
#         x, y, w, h = cv2.boundingRect(nonzero_pts)
#         result = result[y:y + h, x:x + w]

#     return result


# def image_query(imgs:list):

#     current_result = None
#     for index in range(1, len(imgs)):
#         if (current_result is None):
#             current_result = imgs[0]
#         new_addition = imgs[index]
#         current_result = merge_two(current_result, new_addition)

#     cv2.imshow('Result', current_result)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     return

# def image_query_vibe(imgs: list):
#     print("Vibintg")
#     if not imgs:
#         return None

#     current_level = imgs
#     while len(current_level) > 1:
#         next_level = []
#         # merge neighbors two at a time
#         for i in range(0, len(current_level) - 1, 2):
#             merged = merge_two(current_level[i], current_level[i + 1])
#             next_level.append(merged)
#         # odd one out gets carried over untouched to the next round
#         if len(current_level) % 2 == 1:
#             next_level.append(current_level[-1])
#         current_level = next_level

#     result = current_level[0]

#     cv2.imshow('Result', result)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     return result


def main():
    imgs = load_images()
    print(f"Loaded {len(imgs)} images")
    # image_query(imgs)

    

    return


if __name__ == "__main__":
    main()
    
