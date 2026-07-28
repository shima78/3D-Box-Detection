"""
CLI entry point for the box detection pipeline.
"""

import os
import sys
import argparse

# Ensure package imports work both locally and from parent
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from boxdetect.config import CFG
from boxdetect.pipeline import BoxDetectionPipeline


def process_one(path):
    """Process a single .mat file."""
    print("=" * 60)
    print(f"Processing: {os.path.basename(path)}")
    print("=" * 60)
    try:
        pipeline = BoxDetectionPipeline(path)
        pipeline.run()
    except Exception as e:
        print("Error:", e)


def process_all(data_dir):
    """Process all .mat files in a directory."""
    if not os.path.isdir(data_dir):
        print("No data dir:", data_dir)
        return
    mats = [f for f in os.listdir(data_dir) if f.endswith(".mat")]
    for m in mats:
        process_one(os.path.join(data_dir, m))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Box Detection and Measurement")
    parser.add_argument("--file", type=str, help="Process a single .mat file")
    parser.add_argument("--data", type=str, default=CFG.DATA_DIR)
    args = parser.parse_args()

    CFG.ensure_dirs()
    if args.file:
        process_one(args.file)
    else:
        process_all(args.data)
