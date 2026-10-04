"""Download the Amazon Reviews 2023 files for one category into data/raw/."""

from pathlib import Path
from urllib.request import urlretrieve

CATEGORY = "Beauty_and_Personal_Care"
BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw"
RAW_DIR = Path("data/raw")

FILES = {
    f"{CATEGORY}.jsonl.gz": f"{BASE_URL}/review_categories/{CATEGORY}.jsonl.gz",
    f"meta_{CATEGORY}.jsonl.gz": f"{BASE_URL}/meta_categories/meta_{CATEGORY}.jsonl.gz",
}


def download_file(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"Skipping {dest.name} (already downloaded)")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {dest.name} ...")
    tmp = dest.with_name(dest.name + ".part")
    urlretrieve(url, tmp)
    tmp.rename(dest)
    print(f"Done: {dest}")


def main() -> None:
    for filename, url in FILES.items():
        download_file(url, RAW_DIR / filename)


if __name__ == "__main__":
    main()
