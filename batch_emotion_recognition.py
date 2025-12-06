import os
import subprocess
import sys
from pathlib import Path

def find_video_folders(base_path):
    """
    Find all folders named 'videos' in the results_emotions directory structure.
    Expected structure: results_emotions/direction/emotion/level/id/model/processed_date/id/videos
    """
    video_folders = []
    base_path = Path(base_path)
    
    if not base_path.exists():
        print(f"Error: Base path {base_path} does not exist")
        return video_folders
    
    # Walk through all subdirectories to find 'videos' folders
    for root, dirs, files in os.walk(base_path):
        if 'videos' in dirs:
            video_folder = Path(root) / 'videos'
            video_folders.append(video_folder)
    
    return sorted(video_folders)

def extract_path_info(input_folder):
    """
    Extract direction, emotion, level from the input folder path
    Expected structure: results_emotions/direction/emotion/level/id/model/processed_date/id/videos
    """
    path_parts = input_folder.parts
    
    try:
        results_idx = path_parts.index('results_emotions')
        if len(path_parts) > results_idx + 3:
            direction = path_parts[results_idx + 1]  # front, down, etc.
            emotion = path_parts[results_idx + 2]    # neutral, sad, etc.
            level = path_parts[results_idx + 3]      # level_1, level_2, etc.
            return direction, emotion, level
    except (ValueError, IndexError):
        pass
    
    # Fallback: use generic names
    return "unknown_direction", "unknown_emotion", "unknown_level"

def run_emotion_recognition(input_folder, output_base, model_type="3dmm", model_name="EMOCA-emorec"):
    """
    Run the emotion recognition command on a single video folder
    """
    # Extract path information
    direction, emotion, level = extract_path_info(input_folder)
    
    # Create output folder: result_output/direction/emotion/level
    output_folder = Path(output_base) / direction / emotion / level
    
    # Create the output directory if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # IMPORTANT FIX: Convert paths to absolute paths
    current_dir = Path.cwd()  # Get current working directory
    abs_input_folder = current_dir / input_folder
    abs_output_folder = current_dir / output_folder
    
    # Construct the command with absolute paths
    cmd = [
        "python", "demos/test_emotion_recognition_on_images.py",
        "--input_folder", str(abs_input_folder),
        "--output_folder", str(abs_output_folder),
        "--model_type", model_type,
        "--model_name", model_name
    ]
    
    print(f"\n{'='*80}")
    print(f"Processing: {input_folder}")
    print(f"Output to: {output_folder}")
    print(f"Structure: {direction}/{emotion}/{level}")
    print(f"Absolute input: {abs_input_folder}")
    print(f"Absolute output: {abs_output_folder}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}")
    
    try:
        # Change to the EmotionRecognition directory to run the demo
        emotion_recognition_dir = Path("inferno_apps/EmotionRecognition")
        
        # Run the command
        result = subprocess.run(cmd, cwd=emotion_recognition_dir, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ SUCCESS: {direction}/{emotion}/{level}")
            if result.stdout:
                # Print last few lines of stdout for confirmation
                stdout_lines = result.stdout.strip().split('\n')
                print("Last output lines:")
                for line in stdout_lines[-3:]:
                    if line.strip():
                        print(f"  {line}")
        else:
            print(f"❌ FAILED: {direction}/{emotion}/{level}")
            print("STDERR:", result.stderr[-500:])  # Last 500 chars
            if result.stdout:
                print("STDOUT:", result.stdout[-500:])
                
    except Exception as e:
        print(f"❌ EXCEPTION for {direction}/{emotion}/{level}: {str(e)}")
    
    return result.returncode == 0

def main():
    # Configuration
    results_emotions_path = "results_emotions"
    output_base = "result_output"
    model_type = "3dmm"
    model_name = "EMOCA-emorec"
    
    print("🔍 Searching for video folders...")
    video_folders = find_video_folders(results_emotions_path)
    
    if not video_folders:
        print(f"No 'videos' folders found in {results_emotions_path}")
        return
    
    print(f"Found {len(video_folders)} video folders:")
    for i, folder in enumerate(video_folders, 1):
        direction, emotion, level = extract_path_info(folder)
        print(f"  {i:2d}. {folder} -> {direction}/{emotion}/{level}")
    
    # Ask for confirmation
    response = input(f"\nProcess all {len(video_folders)} folders? [y/N]: ").strip().lower()
    if response not in ['y', 'yes']:
        print("Cancelled.")
        return
    
    # Create base output directory
    Path(output_base).mkdir(parents=True, exist_ok=True)
    
    # Process each folder
    successful = 0
    failed = 0
    
    for i, video_folder in enumerate(video_folders, 1):
        direction, emotion, level = extract_path_info(video_folder)
        print(f"\n📁 Processing {i}/{len(video_folders)}: {direction}/{emotion}/{level}")
        
        if run_emotion_recognition(video_folder, output_base, model_type, model_name):
            successful += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n{'='*80}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Total: {len(video_folders)}")
    print(f"📂 Output structure: {output_base}/direction/emotion/level")

if __name__ == "__main__":
    main()