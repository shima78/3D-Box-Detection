"""
Unified Box Detection Package
Fixes import issues for both local and package execution.
"""

import os
import sys

# Ensure this package path is in sys.path
package_dir = os.path.dirname(os.path.abspath(__file__))
if package_dir not in sys.path:
    sys.path.append(package_dir)

# Re-export all main modules
from boxdetect.config import Config, CFG
from boxdetect.io_utils import DataLoader, Visualizer
from boxdetect.geometry import PlaneFitter
from boxdetect.ransac import RANSACPlaneDetector
from boxdetect.mask_utils import MaskProcessor
from boxdetect.measurements import BoxMeasurer
from boxdetect.pipeline import BoxDetectionPipeline

__all__ = [
    "Config",
    "CFG",
    "DataLoader",
    "Visualizer",
    "PlaneFitter",
    "RANSACPlaneDetector",
    "MaskProcessor",
    "BoxMeasurer",
    "BoxDetectionPipeline",
]
