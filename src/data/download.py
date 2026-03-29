"""Stage 1: Download the SMS Spam Collection dataset from UCI."""

import csv
import io
import os
import zipfile
from urllib.request import urlretrieve


SMS_SPAM_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"


def main():
    os.makedirs("data/raw", exist_ok=True)

    zip_path = "data/raw/smsspamcollection.zip"
    print(f"Downloading SMS Spam Collection from UCI...")
    urlretrieve(SMS_SPAM_URL, zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("SMSSpamCollection") as raw:
            lines = io.TextIOWrapper(raw, encoding="utf-8").readlines()

    with open("data/raw/reviews.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])

        for line in lines:
            line = line.strip()
            if not line:
                continue
            label_str, text = line.split("\t", 1)
            label = 1 if label_str == "spam" else 0
            writer.writerow([text, label])

    os.remove(zip_path)
    print(f"Wrote {len(lines)} messages -> data/raw/reviews.csv")


if __name__ == "__main__":
    main()
