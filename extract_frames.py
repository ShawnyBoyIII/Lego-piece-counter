import cv2
import os
import argparse

def extract_frames(video_path, output_dir, frame_interval=10):
    """
    Extracts frames from a video and saves them as images.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory where the extracted frames will be saved.
        frame_interval (int): Save one frame every `frame_interval` frames.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        return

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video loaded: {video_path}")
    print(f"FPS: {fps:.2f}, Total frames: {total_frames}")
    print(f"Extracting 1 frame every {frame_interval} frames...")

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()

        # If the frame was not grabbed, we have reached the end of the video
        if not ret:
            break

        # Save frame based on the interval
        if frame_count % frame_interval == 0:
            # Construct the output filename
            # Format: video_name_frame_0000.jpg
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            out_filename = os.path.join(output_dir, f"{base_name}_frame_{frame_count:04d}.jpg")

            # Save the image
            cv2.imwrite(out_filename, frame)
            saved_count += 1

            # Print progress every 50 saved frames
            if saved_count % 50 == 0:
                print(f"Saved {saved_count} frames so far...")

        frame_count += 1

    cap.release()
    print(f"\nDone! Successfully extracted {saved_count} frames to '{output_dir}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from a video file to build an image dataset.")
    parser.add_argument("video_path", type=str, help="Path to the input video file.")
    parser.add_argument("--output_dir", type=str, default="dataset_frames", help="Directory to save the extracted frames. Defaults to 'dataset_frames'.")
    parser.add_argument("--interval", type=int, default=10, help="Save one frame every N frames. Defaults to 10 (e.g., if video is 30 FPS, interval=10 saves 3 frames per second).")

    args = parser.parse_args()

    extract_frames(args.video_path, args.output_dir, args.interval)
