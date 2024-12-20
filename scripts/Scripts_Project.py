import cv2
import os
import math
# Object type mapping
object_type_map = {
    0: "Unknown",
    1: "Person",
    2: "Car",
    3: "Other Vehicle",
    4: "Other Object",
    5: "Bike"
}

def parse_annotations(txt_file_path):
    """
    Parse the annotations from the text file into a dictionary.
    Returns a dictionary where the key is the frame number, 
    and the value is a list of bounding box details for that frame.
    """
    annotations = {}
    with open(txt_file_path, 'r') as file:
        for line in file:
            fields = line.strip().split()
            if len(fields) != 8:
                continue  # Skip malformed lines
            
            object_id, duration, frame_number, x_lt, y_lt, width, height, obj_type = map(int, fields)
            
            # Add bounding box info to the frame number's list
            if frame_number not in annotations:
                annotations[frame_number] = []
            annotations[frame_number].append({
                "bbox": [x_lt, y_lt, width, height],
                "object_id": object_id,
                "object_type": object_type_map[obj_type]
            })
    return annotations

def draw_bounding_boxes(frame, objects):
    """
    Draw bounding boxes on the frame based on parsed annotations.
    """
    for obj in objects:
        x, y, w, h = obj['bbox']
        label = f"ID:{obj['object_id']} Type:{obj['object_type']}"  # Optional label
        color = (0, 255, 0)  # Green color for bounding boxes
        thickness = 2  # Thickness of bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
    return frame

def process_frame(video_path, annotations_path, output_dir, user_frame_number):
    """
    Extract a specific frame, draw bounding boxes, and save it.
    """
    video_id = os.path.splitext(os.path.basename(video_path))[0]

    # Parse annotations
    annotations = parse_annotations(annotations_path)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    # Get total frame count to validate user input
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if user_frame_number >= total_frames:
        print(f"Error: Frame number {user_frame_number} exceeds total frames {total_frames - 1}.")
        cap.release()
        return

    # Read the specific frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, user_frame_number)
    ret, frame = cap.read()

    if ret:
        # Draw bounding boxes if there are annotations for the frame
        if user_frame_number in annotations:
            frame_with_boxes = draw_bounding_boxes(frame, annotations[user_frame_number])
        else:
            print(f"No annotations found for frame {user_frame_number}.")
            frame_with_boxes = frame

        # Save the frame with bounding boxes
        output_path = os.path.join(output_dir, f"{video_id}_frame_{user_frame_number:04d}.jpg")
        cv2.imwrite(output_path, frame_with_boxes)
        print(f"Frame {user_frame_number} processed and saved as: {output_path}")
    else:
        print(f"Error: Could not read frame {user_frame_number}.")

    cap.release()

# Script 2
def get_annotation_line(txt_file_path, frame_number):
    """
    Retrieve all annotation lines for a specific frame number from the annotation file.
    """
    annotation_lines = []
    with open(txt_file_path, 'r') as file:
        for line in file:
            fields = line.strip().split()
            if len(fields) == 8 and int(fields[2]) == frame_number:
                annotation_lines.append(line.strip())
    return annotation_lines

def process_frames(video_path, annotations_path, output_dir, frame_numbers):
    """
    Extract specific frames from the video, save them as PNGs, and save their annotations.
    """
    # Get video ID from file name
    video_id = os.path.splitext(os.path.basename(video_path))[0]

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video opened successfully
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}.")
        return

    # Get total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame_number in frame_numbers:
        if frame_number >= total_frames:
            print(f"Error: Frame number {frame_number} exceeds total frames {total_frames - 1}.")
            continue

        # Set to the desired frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Unable to read frame {frame_number} from video.")
            continue

        # Retrieve annotation lines
        annotation_lines = get_annotation_line(annotations_path, frame_number)

        # Save frame as PNG
        os.makedirs(output_dir, exist_ok=True)
        frame_file_name = f"{video_id}_{frame_number:04d}.png"

        frame_file_path = os.path.join(output_dir, frame_file_name)
        cv2.imwrite(frame_file_path, frame)
        print(f"Saved frame: {frame_file_path}")

        # Save annotations to a text file
        annotation_file_name = f"{video_id}_{frame_number:04d}_annotation.txt"
        annotation_file_path = os.path.join(output_dir, annotation_file_name)
        with open(annotation_file_path, 'w') as annot_file:
            annot_file.write("\n".join(annotation_lines))
        print(f"Saved annotations: {annotation_file_path}")

    cap.release()

def get_video_length(video_path):
    """
    Get the duration of a video file in seconds.
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        float: Duration of the video in seconds
        None: If there's an error or invalid file
    """
    try:
        # Check if file exists
        if not os.path.exists(video_path):
            print(f"Error: File not found at {video_path}")
            return None
            
        # Open the video file
        video = cv2.VideoCapture(video_path)
        
        # Get total number of frames and frames per second
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        
        # Calculate duration
        duration = total_frames / fps
        
        # Release the video capture object
        video.release()
        
        return duration
        
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        return None


videos_path = os.path.join("..", "videos") 
video_files = os.listdir(videos_path)  
frame_output_dir = "../output_frames"
output_frames_images_annotations_dir = "../output_frames_images_annotations"

for video_file in video_files:
    # Construct full path to video file
    full_video_path = os.path.join(videos_path, video_file)
    annotations_path = f"../annotations/{video_file[:-4]}.viratdata.objects.txt"

    if not os.path.exists(annotations_path):
        raise Exception(f"The file does not exist: {annotations_path}")

    # Get user input for frame number
    try:
        user_frame_number = 100
        process_frame(full_video_path, annotations_path, frame_output_dir, user_frame_number)
    except ValueError:
        print("Error: Please enter a valid integer for the frame number.")

    # Get user input for two frame numbers
    try:
        cap = cv2.VideoCapture(full_video_path)
        duration =  int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        delta = math.floor(duration / 19)

        total_frames = [i for i in range(0, duration, delta)]
        print(f"Length of total frames: {len(total_frames)}")

        for i in range(0,20,2):
            frame_number = [total_frames[i], total_frames[i+1]]
            if len(frame_number) != 2:
                print("Error: Please provide exactly two frame numbers.")
            else:
                print(f"Processing frames: {frame_number}")
                process_frames(full_video_path, annotations_path, output_frames_images_annotations_dir, frame_number)
        cap.release()
    except ValueError:
        print("Error: Please enter valid integers for the frame numbers.")
    
