"""General-purpose dataset retrieval experiment.

Downloads common ML datasets and stores them locally for DVC tracking.
Supports sklearn toy datasets and CSV downloads from URLs.
"""

import argparse
import json
import os
from pathlib import Path


def retrieve_sklearn_dataset(name: str, output_dir: Path) -> dict:
    """Download a dataset from sklearn.datasets and save as CSV."""
    from sklearn import datasets

    loaders = {
        "iris": datasets.load_iris,
        "wine": datasets.load_wine,
        "digits": datasets.load_digits,
        "breast_cancer": datasets.load_breast_cancer,
        "diabetes": datasets.load_diabetes,
    }

    if name not in loaders:
        raise ValueError(f"Unknown sklearn dataset: {name}. Choose from: {list(loaders.keys())}")

    data = loaders[name](as_frame=True)
    df = data.frame

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{name}.csv"
    df.to_csv(csv_path, index=False)

    return {
        "name": name,
        "source": "sklearn",
        "samples": len(df),
        "features": len(df.columns) - 1,
        "path": str(csv_path),
    }


def retrieve_url_dataset(url: str, name: str, output_dir: Path) -> dict:
    """Download a CSV dataset from a URL."""
    import urllib.request

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{name}.csv"
    urllib.request.urlretrieve(url, csv_path)

    line_count = sum(1 for _ in open(csv_path)) - 1  # exclude header

    return {
        "name": name,
        "source": url,
        "samples": line_count,
        "path": str(csv_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Retrieve datasets for ML experiments")
    parser.add_argument(
        "--sklearn",
        nargs="+",
        default=["iris", "wine", "digits"],
        help="sklearn datasets to retrieve (default: iris wine digits)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("raw"),
        help="Output directory for downloaded data (default: raw)",
    )
    args = parser.parse_args()

    manifest = []

    for name in args.sklearn:
        print(f"Retrieving sklearn/{name}...")
        info = retrieve_sklearn_dataset(name, args.output_dir)
        manifest.append(info)
        print(f"  -> {info['samples']} samples, {info['features']} features")

    manifest_path = args.output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest written to {manifest_path}")
    print(f"Total datasets: {len(manifest)}")


if __name__ == "__main__":
    main()
