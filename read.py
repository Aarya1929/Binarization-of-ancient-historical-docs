from PIL import Image
import pytesseract

# Install pytesseract and Tesseract OCR engine
# pip install pytesseract
# Make sure to install Tesseract OCR on your system: https://github.com/tesseract-ocr/tesseract

# Set the path to the Tesseract executable (change this to your Tesseract installation path)
pytesseract.pytesseract.tesseract_cmd = r'E:\T.Y. B.Tech. SEM 1\EDI\tesseract.exe'

def extract_text_from_image(image_path):
    # Open the image file
    img = Image.open(image_path)

    # Use pytesseract to do OCR on the image
    text = pytesseract.image_to_string(img)

    return text

# Example usage
image_path = 'bleed through final/10.jpg'
text_from_image = extract_text_from_image(image_path)

# Now you can apply readability scoring algorithms to the extracted text
# For example, you can use the textstat library for readability scoring in Python
# Install it with: pip install textstat
import textstat

readability_score = textstat.flesch_reading_ease(text_from_image)

print(f'Readability Score: {readability_score}')