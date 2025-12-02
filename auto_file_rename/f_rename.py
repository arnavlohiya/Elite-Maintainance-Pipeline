import os
import re, string
from pathlib import Path
from ocr_api import run_ocr, API_KEY 

WHITEBOARD_PATH = Path.cwd() / Path('sample_images/ocr_img1.jpg') # path to the whiteboard image with tag ID 
VIDEO_PATH = Path.cwd() / Path('360_files/GS012395.mp4') # path to dummy video to be renamed

''' in the coming days, this will be changed to take the frame and video used in 
the whiteboard-detecting script to rename the video file '''

def sanitize_filename(text: str) -> str: 
    # cleans OCR text into a safe, readable filename
    clean = re.sub(r'[<>:"/\\|?*\n\r\t]+', '', text) # removes invalid characters and trims whitespace
    clean = re.sub(r'\s+', '_', clean)               # convert whitespace to underscores
    clean = clean.strip(string.punctuation + '_ ')   # trim punctuation chars
    clean = re.sub(r'_+', '_', clean)                # multiple underscores become single one
    clean = clean.upper()                            # uppercase for consistency
    
    if len(clean) > 50:     # limit filename length if necessary
        clean = clean[:50]

    return clean or "unnamed_video"

def extract_tag(image_path: Path) -> str | None:
    # calls OCR on the provided image and returns the cleaned extracted text.
    print(f"Running OCR on image: {image_path}")
    result = run_ocr(filename=image_path, api_key=API_KEY)

    if not result or not result.get("ParsedResults"):
        print("OCR failed or returned no results.")
        return None

    parsed_text = result["ParsedResults"][0]["ParsedText"].strip()
    print(f"OCR Extracted Text (raw): '{parsed_text}'")

    return sanitize_filename(parsed_text)

def rename_video(video_path: Path, new_name: str):
    # renames the given video file to the new name - keeps extension
    if not video_path.exists():
        print(f"Error: video file not found at {video_path}")
        return

    new_path = video_path.with_name(f"{new_name}{video_path.suffix}")
    print(f"Renaming '{video_path.name}' → '{new_path.name}'")
    os.rename(video_path, new_path)
    print("Rename successful!")


def main():
    if API_KEY is None:
        print("Error: OCR API key not found. Please check ocr_api.txt.")
        return

    if not WHITEBOARD_PATH.exists():
        print(f"Error: whiteboard image not found at {WHITEBOARD_PATH}")
        return

    tag_text = extract_tag(WHITEBOARD_PATH)
    if not tag_text:
        print("Could not extract a valid tag from image.")
        return

    rename_video(VIDEO_PATH, tag_text)

if __name__ == "__main__":
    main()
