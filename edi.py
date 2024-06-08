import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define the modify_pixel function
def modify_pixel(image, target_pixel, pixel_increment):
    # Make a copy of the image to avoid modifying the original
    modified_image = image.copy()

    # Get the indices of pixels with the target pixel value
    pixel_indices = np.where(np.all(modified_image == target_pixel, axis=-1))

    # Increase the intensity of the target pixel
    modified_image[pixel_indices] = np.clip(
        modified_image[pixel_indices] + pixel_increment, 0, 255
    )

    return modified_image

# Load an image
image = cv2.imread('stain/10.jpg')

# Plot the histogram
pixel_values = image.flatten()
plt.hist(pixel_values, bins=256, range=[0, 256])
plt.title('Image Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.show()

# Image processing
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
sharpened_image = cv2.addWeighted(image, 1.5, blurred_image, -0.7, 0)
blurred_image = cv2.bilateralFilter(sharpened_image, 15, 100, 100)

# K-means clustering
pixels = blurred_image.reshape((-1, 3))
pixels = np.float32(pixels)
num_clusters = 3
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
_, labels, centers = cv2.kmeans(pixels, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
centers = np.uint8(centers)
segmented_image = centers[labels.flatten()].reshape((image.shape))

# Display original and segmented images
display = [image, segmented_image]
label = ['Original Image', 'K-mean Clustering']
fig = plt.figure(figsize=(12, 10))
for i in range(len(display)):
    fig.add_subplot(1, 2, i + 1)
    plt.imshow(cv2.cvtColor(display[i], cv2.COLOR_BGR2RGB))
    plt.title(label[i])
plt.show()

# Rotation
angle = 270
height, width = segmented_image.shape[:2]
rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
rotated_image = cv2.warpAffine(segmented_image, rotation_matrix, (width, height))

# Display rotated image
display = [rotated_image]
label = ['After edge']
fig = plt.figure(figsize=(20, 20))
for i in range(len(display)):
    fig.add_subplot(1, 2, i + 1)
    plt.imshow(cv2.cvtColor(display[i], cv2.COLOR_BGR2RGB))
    plt.title(label[i])
plt.show()

# Histogram after rotation
pixel_values = rotated_image.flatten()
plt.hist(pixel_values, bins=256, range=[0, 256])
plt.title('Image Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.show()

# Pixel modification
target_pixel = [40, 40, 40]
pixel_increment = 200
modified_image = modify_pixel(rotated_image, target_pixel, pixel_increment)

# Display original and modified images
display = [rotated_image, modified_image]
label = ['Original Image', 'Modified']
fig = plt.figure(figsize=(12, 10))
for i in range(len(display)):
    fig.add_subplot(1, 2, i + 1)
    plt.imshow(cv2.cvtColor(display[i], cv2.COLOR_BGR2RGB))
    plt.title(label[i])
plt.show()

# Histogram after modification
pixel_values = modified_image.flatten()
plt.hist(pixel_values, bins=256, range=[0, 256])
plt.title('Image Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.show()

# Otsu's thresholding
gray_image = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)
hist, bins = np.histogram(gray_image.flatten(), bins=256, range=[0, 256])
hist_normalized = hist / np.sum(hist)
cdf = np.cumsum(hist_normalized)
otsu_threshold_value, _ = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Plot histogram with Otsu's threshold
plt.figure(figsize=(10, 5))
plt.plot(bins[:-1], hist_normalized, color='b', label='Histogram')
plt.axvline(x=otsu_threshold_value, color='r', linestyle='--', label='Otsu Threshold')
plt.title('Histogram with Otsu Threshold')
plt.xlabel('Pixel Value')
plt.ylabel('Normalized Frequency')
plt.legend()
plt.show()

# Thresholding and mask creation
threshold_value = 110
mask = np.where(segmented_image > threshold_value, 190, 0).astype(np.uint8)

# Display original and masked images
display = [segmented_image, mask]
label = ['Image', 'After mask']
fig = plt.figure(figsize=(12, 10))
for i in range(len(display)):
    fig.add_subplot(1, 2, i + 1)
    plt.imshow(cv2.cvtColor(display[i], cv2.COLOR_BGR2RGB))
    plt.title(label[i])
plt.show()

# Contrast adjustment
contrast = 1.5
brightness = 16
adjusted_image = cv2.convertScaleAbs(mask, alpha=contrast, beta=brightness)
blurred_image11 = cv2.GaussianBlur(adjusted_image, (5, 5), 0)
sharpened_image11 = cv2.addWeighted(adjusted_image, 3.0, blurred_image11, -0.1, 0)

# Save the final image
cv2.imwrite('stain_fin/10.jpg', mask)

# Display original and final images
display = [image, sharpened_image11]
label = ['Original Image', 'Mask']
fig = plt.figure(figsize=(12, 10))
for i in range(len(display)):
    fig.add_subplot(1, 2, i + 1)
    plt.imshow(cv2.cvtColor(display[i], cv2.COLOR_BGR2RGB))
    plt.title(label[i])
plt.show()
