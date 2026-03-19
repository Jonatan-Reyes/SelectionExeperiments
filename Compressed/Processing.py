from typing import List

import csv
import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image
import torch
from timm.data import create_transform

from Filters import run_filters
from ModelArch import Model
from Video import VideoCaptureAsync
from config import Config


class Model_Processor:
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        timm_model_arch: str,
        model_checkpoint: str,
    ):
        self.input_size = input_size
        self.num_classes = num_classes
        self.compiled_model = False
        self.model_transform = create_transform(input_size=self.input_size)
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        # LOADING AND COMPILING MODEL
        print("loading model")
        self.model: Model = Model(
            timm_model_arch=timm_model_arch, num_classes=num_classes
        )
        self.model.load_state_dict(
            torch.load(model_checkpoint, map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()
        print("done loading model")

    def process_transform(self, frame: NDArray[np.uint8]) -> torch.Tensor:
        pil_frame: Image.Image = Image.fromarray(frame.astype(np.uint8))  # type: ignore
        transformed_frame: torch.Tensor = self.model_transform(pil_frame)  # type: ignore
        return transformed_frame

    def process(self, frame: NDArray[np.uint8]) -> NDArray[np.float32]:
        with torch.no_grad():
            transformed_frame = (
                self.process_transform(frame).unsqueeze(0).to(self.device)
            )
            model_logits = (
                self.model.forward(transformed_frame)
                .cpu()
                .detach()
                .numpy()  # type: ignore
                .astype(np.float32)
                .flatten()
            )
        return model_logits


class Frame_Processor:
    def __init__(
        self,
        model_processor: Model_Processor,
        input_size: int,
        use_deinterlace: bool,
        medical_screen_crop: bool,
        medical_screen_crop_pixels: int,
        reject_black_frames: bool,
    ):
        self.model_processor = model_processor
        self.input_size = input_size
        self.use_deinterlace = use_deinterlace
        self.medical_screen_crop = medical_screen_crop
        self.medical_screen_crop_pixels = medical_screen_crop_pixels
        self.reject_black_frames = reject_black_frames

    def crop_frame(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Crop frame similarly to EdgeDevice scope cropping.

        This is a direct copy of the crop_frame logic from
        EdgeDevice.display.utils, kept locally to avoid extra
        dependencies.
        """

        # Expect incoming frame as RGB here
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        margin: int = 10

        top_right_sample = gray[0:margin, w - margin : w]
        black_thresh = int(np.max(top_right_sample)) + 1

        top_non_black = np.where(gray[0, :] > black_thresh)[0]
        bottom_non_black = np.where(gray[h - 1, :] > black_thresh)[0]
        if len(top_non_black) == 0 or len(bottom_non_black) == 0:
            return frame

        top_right_x = int(top_non_black[-1])
        bottom_right_x = int(bottom_non_black[-1])

        y1 = h - 1
        y_start = min(max(0, margin), y1)
        y_end = max(y_start, min(y1, h - 1 - margin))
        sample_ys = np.linspace(y_start, y_end, 40, dtype=int)

        left_edges: list[int] = []
        right_edges: list[int] = []
        for y in sample_ys:
            row = gray[y, :]
            non_black = np.where(row > black_thresh)[0]
            if len(non_black) > 0:
                left_edges.append(int(non_black[0]))
                right_edges.append(int(non_black[-1]))

        if not left_edges:
            return frame

        left_x = int(np.median(left_edges))
        right_x = int(np.median(right_edges + [top_right_x, bottom_right_x]))

        if right_x <= left_x:
            return frame

        return frame[:, left_x : right_x + 1]

    def transform_frame(self, frame_bgr: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Replicate EdgeDevice processing: optional fixed crop, BGR→RGB,
        optional deinterlace, then resize to model input size, followed by
        scope cropping.
        """

        # Optional fixed left-crop as in Input_camera.medical_screen_crop
        if self.medical_screen_crop and frame_bgr is not None:
            frame_bgr = frame_bgr[:, self.medical_screen_crop_pixels :]

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if self.use_deinterlace:
            frame_rgb = cv2.resize(
                frame_rgb,
                (frame_rgb.shape[1], frame_rgb.shape[0] // 2),
                interpolation=cv2.INTER_LINEAR,
            )

        # Scope cropping (copied from EdgeDevice display.utils.crop_frame)
        cropped = self.crop_frame(frame_rgb)
        transformed = cv2.resize(cropped, (self.input_size, self.input_size)).astype(
            np.uint8
        )
        return transformed

    def is_suitable_frame(self, frame_bgr: NDArray[np.uint8]) -> bool:
        """Mirror Frame_Processor.check_suitable_frame: reject mostly-black frames."""

        if not self.reject_black_frames:
            return True
        if frame_bgr is None:
            return False
        return bool(np.mean(frame_bgr) >= 5)

    def process(self, frame_bgr: NDArray[np.uint8]) -> NDArray[np.float32] | None:
        if not self.is_suitable_frame(frame_bgr):
            return None

        transformed = self.transform_frame(frame_bgr)
        # EdgeDevice applies filters and model on the same transformed frame
        model_logits = self.model_processor.process(transformed)
        filter_vals = run_filters(transformed)
        return np.concatenate((model_logits, np.asarray(filter_vals, dtype=np.float32)))


def _build_header(num_filters: int, num_classes: int) -> List[str]:
    """Build CSV header: frame_index, logit_*, filter_* (matching EdgeDevice
    ordering of model outputs followed by filter values, minus timing flags)."""

    header: List[str] = ["frame_index"]
    header += [f"logit_{i}" for i in range(num_classes)]
    header += [f"filter_{i}" for i in range(num_filters)]
    return header


def run_video(config: Config) -> None:
    """Run the compressed pipeline on a single video and write predictions to CSV.

    Each row in the CSV corresponds to one frame:
    frame_index, filter_0..N, logit_0..num_classes-1
    """

    # Set up video reader
    cap = VideoCaptureAsync(config.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {config.video_path}")

    cap.start()

    # Set up model and frame processor
    model_processor = Model_Processor(
        input_size=config.input_size,
        num_classes=config.num_classes,
        timm_model_arch=config.timm_model_arch,
        model_checkpoint=config.model_checkpoint,
    )
    frame_processor = Frame_Processor(
        model_processor,
        input_size=config.input_size,
        use_deinterlace=config.use_deinterlace,
        medical_screen_crop=config.medical_screen_crop,
        medical_screen_crop_pixels=config.medical_screen_crop_pixels,
        reject_black_frames=config.reject_black_frames,
    )

    num_filters = 2  # run_filters returns [blur_val, color_val]
    header = _build_header(num_filters=num_filters, num_classes=config.num_classes)

    frame_index = 0

    with open(config.csv_output_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            combined = frame_processor.process(frame)
            if combined is not None:
                row = [frame_index] + combined.astype(float).tolist()
                writer.writerow(row)

            frame_index += 1

    cap.stop()
    print(f"Finished processing {frame_index} frames.")
    print(f"Predictions written to: {config.csv_output_path}")


if __name__ == "__main__":
    print("Starting compressed video processing pipeline...")
    cfg = Config()
    run_video(cfg)
