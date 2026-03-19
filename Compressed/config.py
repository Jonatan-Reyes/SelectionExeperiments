from dataclasses import dataclass


@dataclass
class Config:
    """Configuration for the compressed video-processing pipeline.

    Adjust the paths and parameters below as needed.
    """

    # Path to the input video file
    video_path: str = (
        "/home/jonatan/Desktop/GAINORG/SelectionExeperiments/Data_Erda/anno0.avi"
    )

    # Path to the CSV file where predictions will be written
    csv_output_path: str = "./predictions.csv"

    # Model parameters (kept consistent with EdgeDevice defaults)
    input_size: int = 224
    num_classes: int = 5
    timm_model_arch: str = "convnextv2_atto"
    model_checkpoint: str = "tm_228.pth"

    # Frame-processing options (mirroring important EdgeDevice behavior)
    # Whether to drop clearly black frames (as in check_suitable_frame)
    reject_black_frames: bool = True

    # Whether to apply simple deinterlacing by halving vertical resolution
    use_deinterlace: bool = False

    # Whether to crop away a fixed left margin (similar to medical_screen_crop)
    medical_screen_crop: bool = False
    medical_screen_crop_pixels: int = 200
