# Web Application for Mapping and Depth Classification of Water Bodies

A Streamlit web app that fetches satellite imagery from **Google Earth Engine**, segments water bodies with **YOLOv8**, estimates their area, and classifies water depth (shallow / medium / deep / very deep) from water colour using a **logistic-regression classifier**. It also tracks changes in water area and depth over a range of years.

> B.Tech final-year project, Department of Computer Science and Engineering, Amrita School of Computing, Amrita Vishwa Vidyapeetham, Coimbatore (April 2025).
> Supervisor: Malathi P, Assistant Professor, CSE.
> Authors: Yegneesh Reddy, Nagarjun Krishna, Sourav Pramod, T Yaswanth Avinash.

---

## Features

- **Three input modes**
  - **Coordinates**: latitude / longitude (optional location name)
  - **Lake Name**: resolved to coordinates via the Google Places API
  - **Upload Image**: analyse your own satellite image (JPG / PNG)
- **Satellite retrieval** from Google Earth Engine (Sentinel-2 RGB), with an adjustable buffer radius (1,000 to 10,000 m).
- **Water segmentation** with YOLOv8 (confidence threshold 0.45), shown as a blue mask over the original image.
- **Area estimation** from the number of masked pixels.
- **Depth classification** from the mean RGB of the central 5% patch of the image, standardised and passed to the trained classifier.
- **Year-by-year analysis** (2015 to current year) with one tab per year.
- **Historical charts** (Plotly) for water area and depth over time, plus **CSV export**.
- **Interactive location map** (geemap / folium satellite basemap).
- Optional export of the retrieved Earth Engine images as GeoTIFFs.

## How it works

```
User input (coords / lake name / image)
        │
        ▼
Google Earth Engine ──► satellite image (per year)
        │
        ├──► YOLOv8 segmentation ──► water mask + area
        │
        └──► Central 5% crop ► mean RGB ► StandardScaler ► classifier ► depth class
                                                   │
                                                   ▼
                          Streamlit dashboard: images, metrics, charts, CSV
```

## Project structure

```
.
├── app.py               # Streamlit application (main entry point)
├── requirements.txt     # Pinned dependencies
├── .env.example         # Template for environment variables
├── .env                 # Your local configuration (not committed)
├── .gitignore           # Excludes .env and virtual environments
├── README.md
├── models/              # Trained model files
│   ├── water_segmentation.pt
│   ├── depth_classifier.h5
│   ├── scaler.pkl
│   └── encoder.pkl
├── data/
│   └── sample_water.jpg # Sample satellite image for the "Upload Image" mode
└── docs/
    ├── project-inventory.md
    └── cleanup-plan.md
```

> Local virtual-environment folders (`venv/`, `myenv/`, `lakeenv/`, `yolo_env/`, `.venv/`, etc.) are **not** part of the project and must not be committed. They are excluded via `.gitignore`.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.11 or newer** | The pinned `numpy` and `pandas` versions in `requirements.txt` need a recent Python. The project report used 3.10, but that will not satisfy these pins. |
| **Google Earth Engine account** | Sign up at <https://earthengine.google.com/> and have a Google Cloud project registered for Earth Engine. |
| **Google Maps Platform API key** | Only needed for the **Lake Name** input (Places API, "Find Place from Text"). |
| **Trained model files** | See [Model files](#model-files). |

## Setup

### 1. Get the code

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` installs:

| Package | Purpose |
|---|---|
| `streamlit` | Web interface |
| `earthengine-api`, `geemap` | Satellite imagery and map widget |
| `ultralytics`, `torch`, `opencv-python` | YOLOv8 segmentation and mask drawing |
| `tensorflow`, `joblib`, `scikit-learn` | Depth classifier, scaler and label encoder |
| `numpy`, `pandas`, `Pillow` | Data and image handling |
| `plotly` | Historical charts |
| `requests` | Google Places API calls and image download |
| `python-dotenv` | Loads configuration from `.env` |

> `torch` and `tensorflow` are large downloads. Allow some time and several GB of disk space.

### 4. Authenticate Google Earth Engine

```bash
earthengine authenticate
earthengine set_project <your-google-cloud-project-id>
```

The app calls `ee.Initialize()` on start-up. If this fails, an error banner appears asking you to run `earthengine authenticate`.

### 5. Configure environment variables

Copy the provided template and fill in your values:

```bash
# Windows
copy .env.example .env
# macOS / Linux
cp .env.example .env
```

Your `.env` should look like this:

```env
# Required only for the "Lake Name" input
GOOGLE_MAPS_API_KEY=your_google_maps_api_key

# Optional: override default model locations
# (relative paths are resolved from the project root)
# YOLO_MODEL_PATH=models/water_segmentation.pt
# DEPTH_MODEL_PATH=models/depth_classifier.h5
# SCALER_PATH=models/scaler.pkl
# ENCODER_PATH=models/encoder.pkl
```

Do not commit `.env`; add it to `.gitignore`.

### Model files

The app loads four files at start-up. By default it looks in `models/`:

| File | Description |
|---|---|
| `water_segmentation.pt` | YOLOv8 segmentation weights, trained on the Roboflow water-body dataset (4,922 images, 256×256) |
| `depth_classifier.h5` | Keras model for depth classification from mean RGB |
| `scaler.pkl` | Fitted scikit-learn `StandardScaler` |
| `encoder.pkl` | Fitted scikit-learn `LabelEncoder` |

Place them in `models/` or point the matching environment variables at them. If any file is missing, the app shows a "Error loading models" message.

### 6. Run the app

```bash
streamlit run app.py
```

Open <http://localhost:8501> in your browser.

---

## Usage

1. Choose an **Input Type** in the sidebar. To try the app quickly, select **Upload Image** and use `data/sample_water.jpg`.
2. Provide the location (coordinates or lake name) or upload an image.
3. For coordinates and lake names, set the **buffer radius** and the **start and end year**.
4. Click **Analyze**.
5. Review, per year: the satellite image, estimated depth class, estimated area, and the segmented water mask.
6. When more than one year is selected, view the area and depth trend charts and download the data as CSV.

## Models and data (from the project report)

**Segmentation: YOLOv8**
- Dataset: Roboflow Universe, 4,922 images (4,350 train / 314 val / 258 test), resized to 256×256
- Reported results: precision 68.1%, recall 87.1%, F1 76.5%, mAP@0.5 = 0.852

**Depth classification**
- Dataset: 152 manually curated Google Earth Engine images, 38 per class, labelled with GEBCO-style depth ranges: Shallow (0–5 m), Medium (5–15 m), Deep (15–50 m), Very Deep (>50 m)
- Features: mean RGB of the central 5% crop, standardised with `StandardScaler`
- Reported accuracy: 100% on the held-out 20% split. The report itself notes this small, well-separated test set may indicate overfitting, so treat depth outputs as indicative rather than survey-grade.

## Known limitations

- **Area units:** area is computed as `masked pixels × PIXEL_TO_METER_SCALE` (currently `0.5`). This is a placeholder factor, not a true ground-sampling-distance conversion, so the values labelled "sq. meters" are approximate. Calibrate it to your image resolution (and square it for area) for accurate results.
- **Depth class labels:** `estimate_depth()` maps class indices to labels with a hard-coded dictionary (including a `Shallow-4` entry). Make sure it matches the class order in your `encoder.pkl`.
- **Depth from colour** is sensitive to turbidity, shadows, cloud cover and season.
- **Earth Engine collection:** the app uses `COPERNICUS/S2`, which Google has marked as deprecated in favour of `COPERNICUS/S2_SR_HARMONIZED`. If imagery stops loading, switch the collection name in `get_satellite_image()`.
- The detection confidence in code is `0.45`.

## Troubleshooting

| Problem | Fix |
|---|---|
| "Google Earth Engine authentication failed" | Run `earthengine authenticate` and `earthengine set_project <project-id>`, then restart the app |
| "Error loading models" | Check that all four files exist in `models/` or that the `.env` paths are correct |
| "Location not found!" | Check the lake name and that `GOOGLE_MAPS_API_KEY` is set and has the Places API enabled |
| "No satellite image found for <year>" | No Sentinel-2 scene exists for that area and year; try a different year or a larger buffer |
| Slow first start | TensorFlow and PyTorch load on first run; model loading is cached afterwards |
| Dependency install errors | Use a fresh virtual environment with Python 3.11+ |

## Future work

Multi-sensor fusion (drone / LiDAR / in-situ), transformer-based models, quantitative bathymetric regression instead of depth classes, forecasting of water levels, and a mobile-friendly offline interface.

## Citation

> Y. Avinash T, S. Pramod, N. Krishna, Y. Reddy, M. P., and S. B. A., "Web Application for Mapping and Depth Classification of Water Bodies," RITEEC 2025 (submitted).

## Acknowledgements

Google Earth Engine, Ultralytics YOLOv8, Roboflow Universe, Streamlit, and the Department of CSE, Amrita School of Computing, Coimbatore.
