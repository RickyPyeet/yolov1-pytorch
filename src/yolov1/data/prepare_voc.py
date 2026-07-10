import csv
import shutil
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
import time
import requests

from tqdm.auto import tqdm
from pathlib import Path

VOC_URLS = {
    "voc2007_trainval": "https://huggingface.co/datasets/JimmyUnleashed/Pascal_VOC/resolve/main/VOCtrainval_06-Nov-2007.tar",
    "voc2012_trainval": "https://huggingface.co/datasets/JimmyUnleashed/Pascal_VOC/resolve/main/VOCtrainval_11-May-2012.tar",
    "voc2007_test": "https://huggingface.co/datasets/JimmyUnleashed/Pascal_VOC/resolve/main/VOCtest_06-Nov-2007.tar"}


def download_file(url: str, destination: Path, retries: int = 3, chunk_size: int = 1024 * 1024):
    """Download a file with retries and a progress bar."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = destination.with_suffix(destination.suffix + ".part")

    if destination.exists():
        print(f"[INFO] {destination.name} already exists. Skipping.")
        return

    for attempt in range(1, retries + 1):
        try:
            print(f"[INFO] Downloading {destination.name} attempt {attempt}/{retries}...")

            with requests.get(url, stream=True, timeout=(30, 120)) as response:

                response.raise_for_status()

                total_size = int(response.headers.get("content-length", 0))

                downloaded_size = 0

                with temporary_path.open("wb") as file:
                    with tqdm(
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=destination.name) as progress_bar:

                        for chunk in response.iter_content(chunk_size=chunk_size):
                            if not chunk:
                                continue

                            file.write(chunk)

                            chunk_length = len(chunk)
                            downloaded_size += chunk_length
                            progress_bar.update(chunk_length)

            if total_size and downloaded_size != total_size:
                raise IOError(f"Incomplete download: received {downloaded_size} of {total_size} bytes.")

            temporary_path.replace(destination)

            print(f"[INFO] Downloaded {destination.name}.")
            return

        except (requests.RequestException, OSError) as error:
            if temporary_path.exists():
                temporary_path.unlink()

            if attempt == retries:
                raise RuntimeError(f"Failed to download {destination.name} after {retries} attempts."
                ) from error

            print(f"[WARNING] Download failed: {error}")
            print("[INFO] Retrying in 3 seconds...")
            time.sleep(3)


def extract_archive(archive_path: Path, destination: Path):
    """Extract a TAR archive."""
    print(f"[INFO] Extracting {archive_path.name}...")

    destination.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, "r") as archive:
        archive.extractall(destination)


def move_directory(source: Path, destination: Path):
    """Move a directory, replacing the destination if necessary."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        shutil.rmtree(destination)

    shutil.move(str(source), str(destination))


def prepare_voc_directories(data_dir: Path):
    """Download and organize Pascal VOC 2007/2012 datasets."""
    archives_dir = data_dir / "archives"
    extraction_dir = data_dir / "_extracted"

    archives_dir.mkdir(parents=True, exist_ok=True)
    extraction_dir.mkdir(parents=True, exist_ok=True)

    archive_paths = {name: archives_dir / Path(url).name for name, url in VOC_URLS.items()}

    for name, url in VOC_URLS.items():
        download_file(url, archive_paths[name])

    # VOC 2007 train/validation
    voc2007_extract_dir = extraction_dir / "voc2007_trainval"
    extract_archive(archive_paths["voc2007_trainval"], voc2007_extract_dir)
    move_directory(voc2007_extract_dir / "VOCdevkit" / "VOC2007", data_dir / "VOCdevkit" / "VOC2007")

    # VOC 2012 train/validation
    voc2012_extract_dir = extraction_dir / "voc2012_trainval"

    extract_archive(archive_paths["voc2012_trainval"], voc2012_extract_dir)
    move_directory(voc2012_extract_dir / "VOCdevkit" / "VOC2012", data_dir / "VOCdevkit" / "VOC2012")

    # VOC 2007 test
    voc2007_test_extract_dir = extraction_dir / "voc2007_test"
    extract_archive(archive_paths["voc2007_test"], voc2007_test_extract_dir)
    move_directory(voc2007_test_extract_dir / "VOCdevkit" / "VOC2007", data_dir / "VOCdevkit_test")

    # Remove the extraction directory
    shutil.rmtree(extraction_dir)

    print("[INFO] Pascal VOC directories prepared.")


def annotations_to_csv(annotation_dirs: list[Path],
                       csv_path: Path):
    """Convert Pascal VOC XML annotations into one CSV file."""

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow(["img_path", "label", "img_width", "img_height", "xmin", "ymin", "xmax", "ymax"])

        for annotation_dir in annotation_dirs:
            for annotation_path in sorted(annotation_dir.glob("*.xml")):
                root = ET.parse(annotation_path).getroot()

                image_path = (annotation_path.parents[1]/"JPEGImages"/f"{annotation_path.stem}.jpg")

                image_width = int(root.findtext("size/width"))
                image_height = int(root.findtext("size/height"))

                for obj in root.findall("object"):
                    difficult = obj.findtext("difficult", default="0")

                    if difficult != "0":
                        continue

                    bbox = obj.find("bndbox")

                    writer.writerow([
                            image_path,
                            obj.findtext("name"),
                            image_width,
                            image_height,
                            float(bbox.findtext("xmin")),
                            float(bbox.findtext("ymin")),
                            float(bbox.findtext("xmax")),
                            float(bbox.findtext("ymax"))])

    print(f"[INFO] Created {csv_path}")


def create_csv_files(data_dir: Path):
    """Create the training and test annotation CSV files."""
    annotations_to_csv(annotation_dirs = [data_dir / "VOCdevkit" / "VOC2007" / "Annotations", data_dir / "VOCdevkit" / "VOC2012" / "Annotations"], 
                       csv_path=data_dir / "voc_2007_2012_trainval.csv")

    annotations_to_csv(annotation_dirs = [ data_dir / "VOCdevkit_test" / "Annotations"],
                       csv_path = data_dir / "voc_2007_test.csv")
    
def prepare_pascal_voc(data_dir: str | Path = "data"):
    """Download and prepare Pascal VOC for YOLOv1."""
    
    data_dir = Path(data_dir)

    prepare_voc_directories(data_dir)
    create_csv_files(data_dir)

    print("[INFO] Pascal VOC preparation completed.")
