import numpy as np
import cv2
from numpy.typing import NDArray
from typing import List


def color_filter(frame: NDArray[np.uint8]) -> np.float32:
    """
    Passing the filter means the frame should be kept
    """
    average_color_row = np.average(frame, axis=0)
    average_color = np.average(average_color_row, axis=0)
    average_color = average_color / sum(average_color)
    return average_color[0]  # Since we want R channel and it follows the RGB format


def blur_filter(frame: NDArray[np.uint8], size: int = 60) -> np.float32:
    """
    Passing the filter means the frame should be kept
    """
    # grab the dimensions of the image and use the dimensions to
    # derive the center (x, y)-coordinates
    orig: NDArray[np.float32] = cv2.resize(
        frame, (500, int(frame.shape[0] * 500 / frame.shape[1]))
    ).astype(np.float32)
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    (h, w) = gray.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))
    # compute the FFT to find the frequency transform, then shift
    # the zero frequency component (i.e., DC component located at
    # the top-left corner) to the center where it will be more
    # easy to analyze
    fft = np.fft.fft2(gray)
    fftShift = np.fft.fftshift(fft)
    fftShift[cY - size : cY + size, cX - size : cX + size] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)
    magnitude = 20 * np.log(np.abs(recon))
    mean = np.mean(magnitude)
    # the image will be considered "blurry" if the mean value of the
    # magnitudes is less than the threshold value
    return mean


def run_filters(frame: NDArray[np.uint8]) -> List[np.float32]:
    blur_val = blur_filter(frame)
    color_val = color_filter(frame)
    filter_val = [blur_val, color_val]
    return filter_val
