"""Provide entity to extract lane lines from an image."""

import math
import cv2
import numpy as np
from abstract_image_process import AbstractImageProcessor


class LaneLineDetector(AbstractImageProcessor):
    """Representation of entity which detects lane lines."""

    def __init__(self, image: np.ndarray):
        """Construct LLD object."""
        self._height, self._width, _ = image.shape
        self._image = image

    def _get_region_of_interest(self, image: np.ndarray, vertices: list):
        """Return only ROI for a given image."""
        masked_image = None

        mask = np.zeros_like(image)
        match_mask_color = 255
        cv2.fillPoly(mask, vertices, match_mask_color)
        masked_image = cv2.bitwise_and(image, mask)

        return masked_image

    def _render_lines(
            self,
            img: np.ndarray,
            lines: np.ndarray,
            color: list = [255, 0, 0],
            thickness: int = 3) -> np.ndarray:
        """Render the detect lines on the original image."""
        rendered_image = None

        line_img = np.zeros(
            (
                img.shape[0],
                img.shape[1],
                3
            ),
            dtype=np.uint8
        )
        rendered_image = np.copy(img)
        if lines is None:
            return
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
        rendered_image = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)

        return rendered_image

    def process(self):
        """Extract line lanes from a given image."""
        region_of_interest_vertices = [
            (0, self._height),
            (self._width / 2, self._height / 2),
            (self._width, self._height),
        ]

        gray_image = cv2.cvtColor(self._image, cv2.COLOR_RGB2GRAY)
        cannyed_image = cv2.Canny(gray_image, 100, 200)

        cropped_image = self._get_region_of_interest(
            cannyed_image,
            np.array(
                [region_of_interest_vertices],
                np.int32
            ),
        )

        lines = cv2.HoughLinesP(
            cropped_image,
            rho=6,
            theta=np.pi / 60,
            threshold=160,
            lines=np.array([]),
            minLineLength=40,
            maxLineGap=25
        )

        left_line_x = []
        left_line_y = []
        right_line_x = []
        right_line_y = []

        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = (y2 - y1) / (x2 - x1)
                if math.fabs(slope) < 0.5:
                    continue
                if slope <= 0:
                    left_line_x.extend([x1, x2])
                    left_line_y.extend([y1, y2])
                else:
                    right_line_x.extend([x1, x2])
                    right_line_y.extend([y1, y2])

        min_y = int(self._height * (3 / 5))
        max_y = int(self._height)

        poly_left = np.poly1d(np.polyfit(
            left_line_y,
            left_line_x,
            deg=1
        ))

        left_x_start = int(poly_left(max_y))
        left_x_end = int(poly_left(min_y))

        poly_right = np.poly1d(np.polyfit(
            right_line_y,
            right_line_x,
            deg=1
        ))

        right_x_start = int(poly_right(max_y))
        right_x_end = int(poly_right(min_y))

        line_image = self._render_lines(
            self._image,
            [[
                [left_x_start, max_y, left_x_end, min_y],
                [right_x_start, max_y, right_x_end, min_y],
            ]],
            thickness=5,
        )

        return line_image
