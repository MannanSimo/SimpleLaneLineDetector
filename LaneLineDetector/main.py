from functools import partial
import numpy as np
import cv2
from moviepy.editor import VideoFileClip
from functools import partial
from abstract_image_process import AbstractImageProcessor
from lane_line_detector import LaneLineDetector


def process_image(
        image: np.ndarray,
        processor: AbstractImageProcessor) -> np.ndarray:
    """Abstract factory for image processors."""
    specific_processor = processor(image)
    return specific_processor.process()


def main():
    """Encapsulate main script workflow."""
    detect_lane_lines = partial(process_image, processor=LaneLineDetector)
    video_output = '..\\solidWhiteRight_output.mp4'
    video_input = VideoFileClip("..\\test_videos\\solidWhiteRight.mp4")
    video_rendered = video_input.fl_image(detect_lane_lines)
    video_rendered.write_videofile(video_output, audio=False)


if __name__ == "__main__":
    main()
