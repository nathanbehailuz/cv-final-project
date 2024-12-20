import os
import shutil

source_dir = "../output_frames_images_annotations"
base_dest_dir = "folder"

# Create the base destination directory if it doesn't exist
os.makedirs(base_dest_dir, exist_ok=True)

# Get all files and sort them
files = sorted(os.listdir(source_dir))

# Helper function to get frame number from filename
def get_frame_number(filename):
    return filename.split('_')[-1].split('.')[0]

# Process files
processed_frames = set()
for file in files:
    if file.endswith('.png') and file not in processed_frames:
        frame1 = file
        frame1_num = get_frame_number(frame1)
        
        # Find the next frame
        base_name = '_'.join(frame1.split('_')[:-1])
        for potential_frame2 in files:
            if potential_frame2.startswith(base_name) and potential_frame2 != frame1:
                frame2 = potential_frame2
                frame2_num = get_frame_number(frame2)
                
                # Find corresponding annotation files
                annot1 = f"{frame1[:-4]}_annotation.txt"
                annot2 = f"{frame2[:-4]}_annotation.txt"
                
                if all(f in files for f in [frame1, frame2, annot1, annot2]):
                    # Create folder name
                    folder_name = f"Pair{base_name.replace('VIRAT', '')}_{frame1_num}_{frame2_num}"
                    pair_folder = os.path.join(base_dest_dir, folder_name)
                    os.makedirs(pair_folder, exist_ok=True)
                    
                    # Copy all four files
                    for file_to_copy in [frame1, frame2, annot1, annot2]:
                        source_file = os.path.join(source_dir, file_to_copy)
                        dest_file = os.path.join(pair_folder, file_to_copy)
                        shutil.copy2(source_file, dest_file)
                    
                    # Mark these frames as processed
                    processed_frames.add(frame1)
                    processed_frames.add(frame2)
                    print(f"Processed pair: {folder_name}")
                    break
        

print("Files have been organized into pairs successfully!")
