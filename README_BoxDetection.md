# 📦 3D Box Detection and Measurement Pipeline

This project implements a modular pipeline to detect and measure 3D boxes from Kinect-style `.mat` point cloud data.  
It uses RANSAC-based plane fitting, mask refinement, and simple geometric measurements to estimate box dimensions.

---

## 📁 Project Structure

```
├── main.py                   # CLI entry point
├── boxdetect/
  ├── data/                     # Input .mat Kinect files
  ├── outputs/                  # Saved plots, masks, and visualizations
  ├── __init__.py               # Package initialization
  ├── config.py                 # Configuration (thresholds, parameters)
  ├── geometry.py               # Plane fitting utilities
  ├── io_utils.py               # Data loading and plotting
  ├── mask_utils.py             # Mask refinement and visualization
  ├── measurements.py           # Box corner detection & dimension computation
  ├── pipeline.py               # Main detection pipeline
  ├── ransac.py                 # RANSAC plane detection logic
  ├── main.py                   # CLI entry point
```

---

## ⚙️ Installation

1. Clone or download the repository.  
2. Install dependencies:
   ```bash
   pip install numpy matplotlib scipy opencv-python
   ```
3. Place your `.mat` files inside the `boxdetect/` directory.  
4. The `outputs/` directory will be created automatically when you run the pipeline.

---

## ▶️ Usage

To process a **single file**:
```bash
python main.py --file path/to/file.mat
```

To process **all .mat files** in the `./` folder:
```bash
python main.py
```

All generated visualizations (point cloud, masks, box dimensions) are saved under `outputs/`.

---

## 🧩 File Overview

| File | Description |
|------|--------------|
| `main.py` | Command-line entry point for running the pipeline. |
| `pipeline.py` | Orchestrates plane detection, masking, and box measurement. |
| `geometry.py` / `ransac.py` | Contain plane fitting and RANSAC logic. |
| `mask_utils.py` | Handles morphological mask refinement. |
| `measurements.py` | Extracts box corners and computes width, length, height. |
| `config.py` | Stores tunable parameters (RANSAC iterations, thresholds). |
| `io_utils.py` | Loads `.mat` Kinect data and saves figures. |

---

## 📏 Algorithm Assumptions

- The scene contains **only one box** placed on a **flat floor plane**.  
- The **top and bottom planes** are assumed to be **parallel**, simplifying height calculation.  
- The **RANSAC distance threshold** (`RANSAC_DIST_THRESH` / `TOP_RANSAC_DIST_THRESH`) controls how tightly points must fit a plane:
  - Too low → may reject valid inliers.
  - Too high → may include outliers or background noise.
- The morphological `OPEN_SIZE` and `CLOSE_SIZE` parameters affect how clean and complete the binary masks are.

---

## ⚡ Notes & Tips

- Increase `RANSAC_ITER` for more stable plane detection (at the cost of speed).  
- Visualize intermediate masks (`Floor Mask`, `Top Plane Mask`) to check segmentation quality.  
- Adjust `SAMPLE_RATE_PC` for faster or denser 3D visualizations.  
- The method is meant for controlled single-object environments (e.g., lab setup).

---

## 🧠 Potential Improvements

- Handle **multiple boxes** or cluttered environments via clustering or region-growing.  
- Replace RANSAC with **MLESAC or Preemptive-RANSAC** for better robustness.  
- Parallelize RANSAC iterations (CPU/GPU).  
- Use **adaptive thresholds** based on local noise.  
- Add **consistency checks** to verify top and bottom planes are parallel.  
- Provide confidence intervals for measured dimensions.


