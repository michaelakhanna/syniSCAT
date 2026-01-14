import numpy as np
import cv2
from tqdm import tqdm


def apply_background_subtraction(signal_frames, reference_frames, params):
    """
    Perform background subtraction and normalize the result to an 8-bit range
    for video encoding.

    The specific subtraction method is selected by
    params["background_subtraction_method"], as defined in the Code Design
    Document:

        - 'reference_frame':
            Computes (Signal - Reference) / Reference on a per-frame basis.

        - 'video_mean':
            Estimates a static background frame from the entire video and
            subtracts this background from each signal frame. In this
            implementation, the background is taken to be the per-pixel
            temporal median of all raw signal frames:

                B(x, y) = median_t Signal_t(x, y)
                Contrast_t(x, y) = Signal_t(x, y) - B(x, y)

            This robustly removes a static background while avoiding the
            "burned-in" negative trails that arise when using a simple
            temporal mean for moving particles.

    After computing the contrast frames, the function performs intensity
    windowing by finding the 0.5 and 99.5 percentile values across the entire
    set of contrast frames, then normalizes each frame to 8-bit [0, 255].

    Args:
        signal_frames (list of np.ndarray): List of raw signal frames
            (typically uint16).
        reference_frames (list of np.ndarray): List of raw reference frames
            (typically uint16). Required for 'reference_frame' method.
        params (dict): Global simulation parameter dictionary (PARAMS). Must
            contain "background_subtraction_method".

    Returns:
        list of np.ndarray: List of 8-bit, normalized frames ready for video
            encoding. Returns an empty list if the inputs are empty.
    """
    if not signal_frames:
        return []

    method = params.get("background_subtraction_method", "reference_frame")

    subtracted_frames = []

    if method == "reference_frame":
        # This preserves the existing behavior:
        # Contrast = (Signal - Reference) / Reference
        if not reference_frames:
            # Cannot perform reference-frame subtraction without references.
            return []

        print("Applying background subtraction using per-frame reference images...")
        for signal_frame, ref_frame in tqdm(
            zip(signal_frames, reference_frames),
            total=len(signal_frames)
        ):
            signal_f = signal_frame.astype(float)
            ref_f = ref_frame.astype(float)
            # Add a small epsilon to the denominator to prevent division by zero.
            subtracted = (signal_f - ref_f) / (ref_f + 1e-9)
            subtracted_frames.append(subtracted)

    elif method == "video_mean":
        # Robust background subtraction from a single video:
        #   1. Build a per-pixel temporal median background frame:
        #        B(x,y) = median_t Signal_t(x,y)
        #   2. Subtract this background from each frame:
        #        Contrast_t(x,y) = Signal_t(x,y) - B(x,y)
        #
        # The reference frames are not used in this method.
        print("Applying background subtraction using temporal median of signal frames (video_mean method)...")

        num_frames = len(signal_frames)
        frame_shape = signal_frames[0].shape

        # Stack all signal frames into a 3D array for per-pixel median calculation.
        # Use float32 to balance precision and memory usage.
        signal_stack = np.empty((num_frames, frame_shape[0], frame_shape[1]), dtype=np.float32)
        for idx, frame in enumerate(signal_frames):
            signal_stack[idx] = frame  # automatic upcast from uint16 to float32

        # Compute the per-pixel temporal median as the background estimate.
        background_frame = np.median(signal_stack, axis=0)

        # Subtract the background from each frame to obtain contrast frames.
        for frame in tqdm(signal_frames, total=num_frames):
            subtracted = frame.astype(np.float32) - background_frame
            subtracted_frames.append(subtracted)

    else:
        raise ValueError(
            f"Unknown background_subtraction_method: {method}. "
            "Supported values are 'reference_frame' and 'video_mean'."
        )

    # Normalize the contrast range across the whole video for consistent brightness.
    # This robustly clips outliers by finding the 0.5 and 99.5 percentile values.
    min_val, max_val = np.percentile(subtracted_frames, [0.5, 99.5])

    final_frames_8bit = []
    if max_val > min_val:
        for frame in subtracted_frames:
            # Scale the frame data to the 0-255 range.
            norm_frame = 255 * (frame - min_val) / (max_val - min_val)
            final_frames_8bit.append(
                np.clip(norm_frame, 0, 255).astype(np.uint8)
            )
    else:
        # Handle the edge case of a video with no contrast.
        final_frames_8bit = [
            np.full(signal_frames[0].shape, 128, dtype=np.uint8)
            for _ in signal_frames
        ]

    return final_frames_8bit


def save_video(filename, frames, fps, size):
    """
    Encodes and saves a list of frames to an .mp4 video file.

    Args:
        filename (str): The output path for the video file.
        frames (list of np.ndarray): A list of 8-bit numpy array frames.
        fps (int): The desired frames per second for the video.
        size (tuple): The (width, height) of the video.
    """
    print(f"Saving final video to {filename}...")
    # Define the codec and create VideoWriter object.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(filename, fourcc, fps, size, isColor=False)

    for frame in frames:
        video_writer.write(frame)

    video_writer.release()
    print("Simulation finished successfully!")