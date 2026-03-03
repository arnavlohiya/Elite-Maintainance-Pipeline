from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import cv2
import torch
import sys
import os

'''
requirements:
torch
transformers
Pillow
opencv-python
'''

# load model and processor 
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# preprocess image 
def preprocess_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not open {path}")
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # enhance contrast
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY, 31, 15)
    
    # Save as temporary cleaned image for debugging if needed
    cleaned_path = "cleaned_temp.png"
    cv2.imwrite(cleaned_path, gray)
    
    return Image.open(cleaned_path).convert("RGB")

def get_img():
    if len(sys.argv) < 2:
        print("Error: No image path provided.")
        print(f"Usage: python {sys.argv[0]} <image_path>")
        sys.exit(1)
        
    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print(f"Error: image file not found at path: '{img_path}'")
        sys.exit(1)
        
    return img_path

# extract HW with OCR 
def extract_handwriting_text(image_path):
    image = preprocess_image(image_path)
    
    pixel_values = processor(images=image, return_tensors="pt").pixel_values # type: ignore
    
    with torch.no_grad():
        generated_ids = model.generate(pixel_values) # type: ignore
    
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()

# entry
if __name__ == "__main__":
    img_path = get_img() # get the image path
    extracted_text = extract_handwriting_text(img_path)
    print("Extracted Text:")
    print(extracted_text)
