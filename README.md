# Fire Radiative Power (FRP) Analysis Workflow

A comprehensive Python workflow for analyzing cumulative Fire Radiative Power (cFRP) patterns across the western US using VIIRS active fire detections and FIREDpy fire perimeter data.

## Overview

This workflow combines VIIRS (Visible Infrared Imaging Radiometer Suite) active fire detections with FIRED (Fire Events Delineation) fire perimeter data to calculate Fire Radiative Power statistics at 375m grid cells spatial resolution.

## Key Features

- **Multi-scale Analysis**: Spatial aggregation from 375m pixels to ecoregion-level statistics
- **Temporal Pattern Detection**: Monthly, seasonal, and interannual trend analysis
- **Sophisticated FRP Metrics**: Cumulative FRP, percentiles (90th, 95th, 97th, 99th), day/night breakdowns
- **Parallel Processing**: Optimized for large-scale datasets with configurable parallelization
- **Ecosystem Analysis**: Ecoregion-specific fire behavior characterization

## Repository Structure

```
FIRED-frp/
├── code/
│   ├── frp_FIRED_v3.py           # Main workflow script
│   ├── frp_data_exp_v1.ipynb     # Comprehensive data exploration notebook
│   ├── __functions.py            # Helper functions
│   └── deprecated/               # Previous versions
├── data/
│   ├── input/
│   │   ├── AFD/VIIRS/           # Active fire detection data
│   │   ├── firePerimeters/FIRED/ # Fire perimeter datasets
│   │   └── AOI/                 # Area of interest boundaries
│   └── output/
│       ├── AFD/                 # Processed fire detection outputs
│       └── firePerimeters/      # Processed fire perimeter outputs
└── README.md
```

## Installation & Requirements

### Python Dependencies
```bash
pip install numpy pandas geopandas matplotlib seaborn rasterio shapely scipy tqdm fiona
```

### Required Data Sources

1. **VIIRS Active Fire Data**: Download from NASA FIRMS and concatenate into `data/input/AFD/VIIRS/VIIRS_cat.csv`
2. **FIRED Fire Perimeters**: Place `fired_conus_ak_2000_to_2025_events_merged.gpkg` in `data/input/firePerimeters/FIRED/`
3. **Area of Interest**: Western US boundaries in `data/input/AOI/westUS_5070.gpkg`
4. **Ecoregions**: WWF ecoregion boundaries for ecosystem analysis

## Quick Start

### 1. Run Main Workflow
```python
python code/frp_FIRED_v3.py
```

### 2. Explore Results
Open and run `code/frp_data_exp_v1.ipynb` for comprehensive data exploration and visualization.

## Workflow Steps

### Core Processing Pipeline

1. **AOI Selection**: Load and process area of interest (default: Western US)
2. **Fire Perimeter Processing**: Filter FIRED perimeters, apply 3km buffer + convex hull
3. **Active Fire Detection Processing**: Load VIIRS data, filter low confidence detections
4. **Spatial Integration**: Join fire detections to perimeters with temporal constraints
5. **Grid Creation**: Generate 375m regular grid for spatial aggregation
6. **FRP Aggregation**: Calculate sophisticated FRP statistics per grid cell
7. **Quality Control**: Filter fires with minimum observation thresholds

### Data Exploration & Analysis

The Jupyter notebook (`frp_data_exp_v1.ipynb`) provides:

- **Temporal Trends**: Monthly and yearly cFRP patterns (2012-2025)
- **Seasonal Analysis**: Peak fire month identification by ecoregion
- **Ecoregion Comparison**: Faceted visualizations with fixed scales
- **Statistical Analysis**: Trend detection with confidence intervals
- **Spatial Visualization**: Maps with integrated bar charts and state boundaries

## Key Outputs

### Primary Data Products

1. **Gridstats File**: `viirs_snpp_jpss1_afd_latlon_firesFIRED_{aoi}_pixar_gridstats.gpkg`
   - 375m resolution grid cells with FRP statistics
   - Cumulative FRP (frp_csum) in W/km²
   - Percentile metrics (P90, P95, P97, P99)
   - Day/night breakdowns
   - Temporal metrics (duration, peak dates)

2. **Processed Fire Perimeters**: Buffered and filtered fire boundaries
3. **Active Fire Pixels**: VIIRS detections as pixel areas with FRP/area calculations

## Configuration Options

### Performance Tuning
```python
ENABLE_PARALLELIZATION = True  # Enable/disable parallel processing
MAX_CORES = 8                  # Maximum cores to use (None = all available)
```

### Analysis Parameters
```python
buffer_dist = 3000            # Fire perimeter buffer distance (meters)
grid_resolution = 375         # Grid cell size (meters)
min_observations = 10         # Minimum fire detections per fire
```

### AOI Selection
```python
aoi_options = {
    'westUS': 'data/input/AOI/westUS_5070.gpkg',
    'srm': 'data/input/AOI/na_cec_eco_l3_srme.gpkg',
    # Add custom AOIs here
}
selected_aoi = 'westUS'  # Change to switch study areas
```

## Methodology

### FRP Calculation
Fire Radiative Power is calculated as:
```
FRP_density = FRP_detection / pixel_area
cumulative_FRP = Σ(FRP_density) per grid cell
```

### Quality Control
- Remove low confidence VIIRS detections
- Apply temporal constraints (±14 days from fire perimeter dates)
- Minimum observation thresholds (≥10 detections per fire)
- Duplicate detection resolution (keep largest fire assignment)

### Statistical Analysis
- Linear regression for trend detection
- Confidence intervals using standard deviation
- Significance testing (p < 0.05)
- Seasonal decomposition and peak identification

## Visualization Features

### Time Series Analysis
- Monthly and yearly trend plots with confidence intervals
- Statistical summaries (R², p-values, annual change rates)
- Peak period identification and quantification

### Spatial Analysis
- Ecoregion maps with integrated monthly bar charts
- Peak fire month mapping with discrete color scales
- State boundary overlays for geographic context
- Faceted comparisons with fixed scales

### Multi-panel Displays
- Combined monthly/yearly/spatial visualization panels
- Top-performing ecoregion focus analysis
- Comparative seasonal pattern assessment

## Performance Characteristics

### Scalability
- **Dataset Size**: >5,000 fires, >9.8M fire detections
- **Processing Time**: ~30-60 minutes for western US (8 cores)
- **Memory Usage**: ~4-8GB for full western US analysis
- **Output Size**: ~500MB-2GB depending on fire density

### Optimization Features
- Chunked parallel processing for memory efficiency
- Spatial indexing for rapid intersection operations
- Configurable core usage for different computing environments
- Progress tracking for long-running operations

## Citation

If you use this workflow in your research, please cite:

```
[WIP]
```

## Contributors

- **Maxwell Cook** (maxwell.cook@colorado.edu): Original workflow development and FRP aggregation methods
- **Nate Hofford** (nate.hofford@colorado.edu): Multi-dataset compatibility, performance optimization, and analysis expansion

## Support

For questions, issues, or contributions:
- Open an issue in the repository
- Contact Nate Hofford (nate.hofford@colorado.edu)
- Refer to the detailed methodology in `frp_data_exp_v1.ipynb`

---

**Note**: This workflow requires substantial computational resources for large-scale analysis!  
