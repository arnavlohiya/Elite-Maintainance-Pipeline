import os
import time # solely for demo
from pathlib import Path
from f_rename import extract_tag, rename_video
from collections import deque

VIDEO_FOLDER_PATH = Path.cwd() / Path('videos')
IMG_PATH = Path.cwd() / Path('images')
to_process = deque()

def init_queue(video_path: Path):
    try:
        for filename in os.listdir(video_path):
            to_process.append(filename)
        print(f'Added all files at {video_path} to processing queue.')
    except Exception as e:
        print(f'Exception encountered in init_queue: {e}')

def extract_whiteboards():
    # placeholder for call to script that extracts single optimal whiteboard
    try:
        for filename in to_process:
            print(f'Processing {filename}')
            # call here
            time.sleep(0.01)
            # save image to path
            # print('Image saved')
        return "Done"
    
    except Exception as e:
        print(f'Exception encountered in extract_whiteboards: {e}')

def run_ocr_rename_pipeline(image_path: Path, video_path: Path):
    path = None
    current_file = None
    try:
        tag_map = {}
        path = image_path
        # run OCR on all whiteboard images 
        for img in path.iterdir():
            current_file = img
            if not img.is_file(): continue
            if img.name.startswith('.'): continue

            stem = img.stem             # 'abc123.jpg' → 'abc123'
            tag_id = extract_tag(img)   # extract tag using OCR

            if tag_id is not None:
                tag_map[stem] = tag_id
            else: print(f"WARNING: No tag extracted from {img.name}")

        if not tag_map:
            print("No valid IDs extracted from any images.")
            return
        
        path = video_path
        # match each extracted tag to its source video 
        for video_file in path.iterdir():
            if not video_file.is_file(): continue
            if video_file.name.startswith('.'): continue

            vid_stem = video_file.stem 
            if vid_stem in tag_map:
                new_name = tag_map[vid_stem]

                print(f"\nMatch found:")
                print(f"  Image Stem: {vid_stem}")
                print(f"  Video:      {video_file.name}")
                print(f"  New Tag:    {new_name}")

                rename_video(video_file, new_name)

        print("\nPipeline complete.\n")
    except FileNotFoundError:
        # check if it was the folder or a specific file
        target = current_file if current_file else path
        print(f"Error: Not found at {target}")
    except PermissionError:
        target = current_file if current_file else path
        print(f"Error: Permission denied for {target}")
    except Exception as e:
        target = current_file if current_file else path
        print(f"Error: Exception in pipeline processing {target}: {e}")

def main():
    if VIDEO_FOLDER_PATH.exists():
        init_queue(VIDEO_FOLDER_PATH)
    extract_whiteboards()
    if IMG_PATH.exists():
        run_ocr_rename_pipeline(IMG_PATH, VIDEO_FOLDER_PATH)
    print('OCR-Rename pipeline completed')

if __name__ == "__main__":
    main()
