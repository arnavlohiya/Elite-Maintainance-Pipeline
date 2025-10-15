from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import cv2
import torch

# --- 1. Load model + processor ---
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# --- 2. Preprocess image ---
def preprocess_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not open {path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Optional: adaptive threshold to enhance contrast
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY, 31, 15)
    
    # Save as temporary cleaned image for debugging if needed
    cleaned_path = "cleaned_temp.png"
    cv2.imwrite(cleaned_path, gray)
    
    return Image.open(cleaned_path).convert("RGB")

# --- 3. OCR function ---
def extract_handwriting_text(image_path):
    image = preprocess_image(image_path)
    
    pixel_values = processor(images=image, return_tensors="pt").pixel_values # type: ignore
    
    with torch.no_grad():
        generated_ids = model.generate(pixel_values) # type: ignore
    
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()

# --- 4. Example usage ---
if __name__ == "__main__":
    img_path = "mhid_pic.png"  # replace with your image file
    extracted_text = extract_handwriting_text(img_path)
    print("📝 Extracted Text:")
    print(extracted_text)
