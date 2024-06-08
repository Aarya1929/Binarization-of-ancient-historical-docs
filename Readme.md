BINARIZATION OF ANCIENT HISTORICAL DOCUMENTS

Introduction

This Python script performs a series of image processing tasks, including histogram plotting, image blurring, sharpening, k-means clustering, rotation, pixel modification, thresholding, contrast adjustment, and finally, text extraction using OCR (Optical Character Recognition) with Tesseract. The extracted text's readability is then scored using the Flesch Reading Ease formula.

Dependencies

Python packages: OpenCV, NumPy, Matplotlib, PIL (Pillow), pytesseract, textstat

Description of Script

Image Loading and Histogram Plotting: Loads an image and plots its histogram.
Image Processing:
Blurring and sharpening.
K-means clustering for segmentation.
Image rotation.
Pixel modification to enhance specific pixel values.
Thresholding and mask creation.
Contrast adjustment.
Saving Processed Image: Saves the final processed image.
Text Extraction and Readability Scoring:
Uses Tesseract OCR to extract text from an image.
Calculates the readability score using the Flesch Reading Ease formula.

Example Output

The script will display several plots showing the original and processed images, histograms, and the segmented image with applied modifications. The final processed image is saved, and the extracted text's readability score is printed.
