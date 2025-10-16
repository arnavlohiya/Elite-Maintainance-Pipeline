import requests
import json
import os, sys

# --- configuration ---
API_KEY = 'K84932287888957' # Diego's API key 

# OCR.space API endpoint
OCR_URL = 'https://api.ocr.space/parse/image'

def ocr_space_file(filename, api_key, language='eng', is_overlay_required=False):
    """
    sends a local image file to the OCR.space API for handwriting recognition.
    """
    
    # payload for the POST request
    payload = {
        'apikey': api_key,
        'language': language,
        'isOverlayRequired': is_overlay_required,
        'detectOrientation': 'true', 
        'scale': 'true', 
        'filetype': 'jpg',
        'OCREngine': 2 
    }
    
    try:
        # open the image file in binary read mode
        with open(filename, 'rb') as f:
            # send the POST request
            response = requests.post(
                OCR_URL, 
                files={filename: f}, 
                data=payload
            )
        
        response_json = json.loads(response.content.decode())
        return response_json
    
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"An error occurred during the API request: {e}")
        return None


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

IMAGE_PATH = get_img()

# --- main entry ---
if __name__ == '__main__':
    print(f"Sending image '{IMAGE_PATH}' to OCR.space...")
    result = ocr_space_file( filename=IMAGE_PATH, api_key=API_KEY )
    
    if result and result.get("ParsedResults"):
        # the OCR result is in the first element of 'ParsedResults'
        parsed_text = result["ParsedResults"][0]["ParsedText"]
        
        # print("\n--- OCR Result (Raw) ---")
        # print(result) # Print the entire JSON response for debugging
        print("\n--- Extracted Text ---")
        print(parsed_text)
        
    elif result and result.get("ErrorMessage"):
        print(f"\n--- API Error ---")
        print(result["ErrorMessage"])
    
    else:
        print("\nCould not get a valid response or no text was detected.")
