# =============================================================================
# make_walkway_gif.py
# NEW SCRIPT: Convert a folder of PNG frames into an animated GIF
# Perfect for visualizing Simulated Annealing progress on campus walkways
# =============================================================================

import os
import glob
from PIL import Image
import re
from datetime import datetime

def create_walkway_evolution_gif(image_folder: str, 
                                 output_filename: str = None,
                                 duration: int = 120,      # milliseconds per frame
                                 loop: int = 0,            # 0 = infinite loop
                                 sort_by_number: bool = True):
    """
    Creates an animated GIF from all .png files in a folder.
    
    TEACHING NOTES (Engineering Design Optimization context):
        In your SA project, you will likely generate many PNG frames 
        (e.g. every 100 iterations showing the current best network).
        This script turns those static frames into a smooth animation 
        so you can clearly see how the walkway layout evolves — 
        nodes appearing/disappearing, paths being pruned, hubs forming, etc.
        
        This is a standard post-processing step when using metaheuristics 
        like Simulated Annealing (Ch. 8.6 in your book).
    """

    # 1. Find all PNG files in the folder
    png_files = glob.glob(os.path.join(image_folder, "*.png"))
    
    if not png_files:
        print(f"❌ No .png files found in folder: {image_folder}")
        return

    print(f"Found {len(png_files)} PNG frames in {image_folder}")

    # 2. Sort the images intelligently
    if sort_by_number:
        # Sort by numbers in the filename (e.g., frame_001.png, frame_002.png, ...)
        def extract_number(filename):
            match = re.search(r'(\d+)', os.path.basename(filename))
            return int(match.group(1)) if match else 0
        png_files.sort(key=extract_number)
    else:
        # Sort alphabetically (default behavior)
        png_files.sort()

    # 3. Open all images
    frames = []
    for png_path in png_files:
        try:
            img = Image.open(png_path)
            # Convert to RGB (required for GIF, removes alpha if present)
            frames.append(img.convert('RGB'))
            print(f"  Loaded: {os.path.basename(png_path)}")
        except Exception as e:
            print(f"  Warning: Could not load {png_path} — {e}")

    if not frames:
        print("❌ No valid frames could be loaded.")
        return

    # 4. Set output filename with timestamp if none provided
    if output_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"campus_walkway_evolution_{timestamp}.gif"
    elif not output_filename.lower().endswith('.gif'):
        output_filename += '.gif'

    # 5. Save as animated GIF
    frames[0].save(
        output_filename,
        save_all=True,
        append_images=frames[1:],      # all frames after the first
        duration=duration,             # time each frame is shown (ms)
        loop=loop,                     # 0 = infinite loop
        optimize=True
    )

    print(f"\n✅ GIF successfully created!")
    print(f"   Output file : {output_filename}")
    print(f"   Total frames: {len(frames)}")
    print(f"   Duration per frame: {duration} ms")


# =============================================================================
# MAIN - Easy to customize for your project
# =============================================================================
if __name__ == "__main__":
    # CHANGE THESE SETTINGS TO MATCH YOUR PROJECT:
    
    IMAGE_FOLDER = r"C:\Users\Alex R. Williams\Documents\School\Optimization\Project\Images"          # Folder containing your PNG frames
    
    create_walkway_evolution_gif(
        image_folder=IMAGE_FOLDER,
        output_filename="campus_walkway_sa_evolution.gif",   # nice descriptive name
        duration=200,                   # 150 ms = reasonably smooth animation
        loop=0,                         # infinite loop (good for presentations)
        sort_by_number=True             # sort by numbers in filename
    )