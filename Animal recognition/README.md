# Isolation Process
- Take original picture while in-flight
- Import into GIMP or similar and adjust image values curve until animal is significantly different from the background texture
- Blur image to reduce pixel noise (large kernel, as GIMP processing also increased noise contrast), Convert to LAB color space, build 2D histogram using a/b channels to get the most prominent color(s) (grass, since most of the picture is covered by it), **get Euclidian distance from background color**.
- Use Otsu thresholding to separate the animal from the background
- Clean mask from Otsu by applying median blur to remove single-pixel speckle, then morph to both remove tiny noise blobs and bridge small gaps between larger blobs (darker parts of animal body)
- Choose animal contour blob by removing remaining small blobs and ignoring improbably large blobs. **Then draw a bounding box around animal on original picture**
