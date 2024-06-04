import os
import cv2

def images_to_video(image_folder, video_output, fps=30):
    # Get the list of image files in the directory
    image_files = sorted([os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith('.jpg') or file.endswith('.png')])
    
    # Read the first image to get dimensions
    first_image = cv2.imread(image_files[0])
    height, width, _ = first_image.shape

    # Initialize video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Choose codec (codec may vary based on your system)
    out = cv2.VideoWriter(video_output, fourcc, fps, (width, height))

    # Loop through image files and write frames to video
    for image_file in image_files:
        frame = cv2.imread(image_file)
        out.write(frame)

    # Release video writer
    out.release()

# Example usage
image_folder = './results0.mp4'  # Directory containing images
video_output = 'outputrissu.mp4'  # Output video file
fps = 30  # Frames per second

images_to_video(image_folder, video_output, fps)
