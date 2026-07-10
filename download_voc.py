import argparse
import sys
from pathlib import Path

from src.yolov1.data.prepare_voc import prepare_pascal_voc

def main():
    parser = argparse.ArgumentParser(description="Download and preprocess Pascal VOC for YOLOv1.")

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory in which Pascal VOC will be stored.")

    args = parser.parse_args()

    prepare_pascal_voc(data_dir = args.data_dir)

    print("[INFO] Pascal VOC download and preparation completed, enjoy! :)")


if __name__ == "__main__":
    main()
