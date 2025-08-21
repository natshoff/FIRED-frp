#!/usr/bin/env python3
"""
FRP Code Workflow Script

Purpose:
Calculate cumulative fire radiative power (cumFRP in W/Km^2) from an active fire detection 
source (AFD), for a specific area of interest (AOI), using a fire perimeter data set to bound detection.

Performance Optimizations:
- Parallelized grid creation using multiprocessing
- Parallelized fire-by-fire FRP aggregation 
- Configurable parallelization settings (ENABLE_PARALLELIZATION, MAX_CORES)
- Optimized for large west-wide datasets

Inputs:
- Active Fire Detection: VIIRS
- Area of interest (AOI): Configurable (e.g., Western US, Southern Rockies)
- Fire Perimeters: FIRED (post-processed 5/11)

Outputs:
- Fire perimeters: {aoi}_{fire_dataset}_fires_{date_range}_filtered.gpkg
- Buffered hulls: {aoi}_{fire_dataset}_fires_{date_range}_buffered.gpkg  
- AFD points: {aoi}_{fire_dataset}_viirs_{date_range}_points.gpkg
- AFD pixels: {aoi}_{fire_dataset}_viirs_{date_range}_pixels.gpkg
- Analysis grid: {aoi}_{fire_dataset}_grid_{date_range}_375m.gpkg
- **Main output**: {aoi}_{fire_dataset}_gridstats_{date_range}_final.gpkg

Author Acknowledgement:
Maxwell Cook (maxwell.cook@colorado.edu) created the original workflow and methods for aggregating FRP
Nate Hofford (nate.hofford@colorado.edu) modified this original workflow to be compatible with various AFDs, AOIs, and fire perimeters
Performance optimizations added for large-scale processing
"""

import sys
import os
import math
import time
import gc
import numpy as np
import pandas as pd
import geopandas as gpd
import fiona
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio as rio
import re
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
from functools import partial
from pathlib import Path
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from rasterio.features import rasterize
from tqdm import tqdm
from shapely.geometry import Point, Polygon, box
from scipy.spatial import cKDTree

# Import custom functions (with fallback if not available)
try:
    from __functions import *
except ImportError:
    print("Warning: Could not import custom functions from __functions.py")


def detect_aoi_name(aoi_filepath):
    """
    Extract AOI name from the file path for consistent naming throughout the workflow.
    
    Parameters:
    aoi_filepath (str): Path to the AOI file
    
    Returns:
    str: AOI name (e.g., 'westUS', 'SRM')
    """
    filename = os.path.basename(aoi_filepath)
    # Remove file extension
    aoi_name = os.path.splitext(filename)[0]
    # Remove common suffixes like projection codes
    aoi_name = re.sub(r'_\d{4,5}$', '', aoi_name)  # Remove _5070, _4326, etc.
    return aoi_name


def create_cell_batch(args):
    """Create a batch of grid cells for parallel processing"""
    x_batch, y_coords, res = args
    cells = []
    for x in x_batch:
        for y in y_coords:
            cells.append(Polygon([(x, y), (x + res, y), (x + res, y + res), (x, y + res)]))
    return cells


def process_single_fire_wrapper(args):
    """
    Process a single fire for FRP aggregation.
    This function will be called in parallel.
    """
    fire_data, afds_pix_subset, grid_subset = args
    
    # Get detections for this fire
    fire_detections = afds_pix_subset[afds_pix_subset['merge_id'] == fire_data['merge_id']]
    
    if len(fire_detections) == 0:
        return None
    
    # Aggregate fire pixels to the grid using the aggregate_frp function
    fire_grid = aggregate_frp_global(fire_detections, grid_subset)
    fire_grid['merge_id'] = fire_data['merge_id']
    
    return fire_grid


def aggregate_frp_global(detections, grid):
    """
    Global version of aggregate_frp function for parallel processing.
    Aggregate FRP statistics for detections within grid cells.
    """
    import time
    
    # Spatial join detections to grid
    overlay = gpd.sjoin(detections, grid[['grid_index', 'geometry']], 
                       how='inner', predicate='intersects')
    
    if len(overlay) == 0:
        return grid[['grid_index', 'geometry']].copy()
    
    # Calculate FRP per area for aggregation
    overlay['frp_fr'] = overlay['frp'] / overlay['pix_area']
    
    # Main aggregation with statistics
    aggregated = overlay.groupby('grid_index').agg(
        afd_count=('obs_id', 'count'),
        frp_csum=('frp_fr', 'sum'),
        frp_max=('frp_fr', 'max'),
        frp_min=('frp_fr', 'min'),
        frp_first=('frp_fr', 'first'),
        frp_p90=('frp_fr', lambda x: x.quantile(0.90) if not x.empty else 0),
        frp_p95=('frp_fr', lambda x: x.quantile(0.95) if not x.empty else 0),
        frp_p97=('frp_fr', lambda x: x.quantile(0.97) if not x.empty else 0),
        frp_p99=('frp_fr', lambda x: x.quantile(0.99) if not x.empty else 0),
        unique_days=('acq_date', 'nunique'),
        first_obs_date=('acq_date', 'min'),
        last_obs_date=('acq_date', 'max')
    ).reset_index()
    
    # Day/night breakdown
    daynight_grouped = overlay.groupby(['grid_index', 'daynight']).agg(
        count=('obs_id', 'count'),
        max=('frp_fr', 'max'),
        sum=('frp_fr', 'sum'),
        first=('frp_fr', 'first'),
        p90=('frp_fr', lambda x: x.quantile(0.90) if not x.empty else 0),
        p95=('frp_fr', lambda x: x.quantile(0.95) if not x.empty else 0),
        p97=('frp_fr', lambda x: x.quantile(0.97) if not x.empty else 0),
        p99=('frp_fr', lambda x: x.quantile(0.99) if not x.empty else 0)
    ).unstack(fill_value=0)
    
    # Flatten column names for day/night stats
    daynight_stats = {}
    for stat in ['count', 'max', 'sum', 'first', 'p90', 'p95', 'p97', 'p99']:
        daynight_stats[f'{stat}_D'] = f'frp_{stat}_day'
        daynight_stats[f'{stat}_N'] = f'frp_{stat}_night'
    
    daynight_grouped.columns = [daynight_stats.get(f'{col[0]}_{col[1]}', f'{col[0]}_{col[1]}') 
                               for col in daynight_grouped.columns]
    daynight_grouped = daynight_grouped.reset_index()
    
    # Merge day/night stats with main aggregation
    aggregated = aggregated.merge(daynight_grouped, on='grid_index', how='left')
    
    # Calculate additional metrics
    aggregated['day_count'] = aggregated.get('frp_count_day', pd.Series([0]*len(aggregated))).fillna(0)
    aggregated['night_count'] = aggregated.get('frp_count_night', pd.Series([0]*len(aggregated))).fillna(0)
    
    # Calculate observation duration in days
    aggregated['obs_duration'] = (aggregated['last_obs_date'] - aggregated['first_obs_date']).dt.days
    
    # Find day of maximum FRP
    max_frp_dates = overlay.loc[overlay.groupby('grid_index')['frp_fr'].idxmax()]
    max_frp_dates = max_frp_dates[['grid_index', 'acq_date']].rename(columns={'acq_date': 'day_max_frp'})
    aggregated = aggregated.merge(max_frp_dates, on='grid_index', how='left')
    
    # Join results back to grid
    grid_result = grid.merge(aggregated, on='grid_index', how='right')
    
    return grid_result


def main():
    print("Starting FRP analysis workflow...")
    
    # =============================================================================
    # Configuration: Set AOI file path here
    # =============================================================================
    
    # Parallelization settings for performance optimization
    ENABLE_PARALLELIZATION = True  # Set to False to disable all parallel processing
    MAX_CORES = 8  # Maximum number of cores to use (None = use all available)
    
    # Default AOI options - uncomment the one you want to use or add your own
    aoi_options = {
        'westUS': 'data/input/AOI/westUS_5070.gpkg',
        'srm': 'data/input/AOI/na_cec_eco_l3_srme.gpkg',
        # Add more AOI options here as needed
        # 'customAOI': 'data/input/AOI/your_custom_aoi.gpkg',
    }
    
    # Set which AOI to use (change this to switch AOIs)
    selected_aoi = 'srm'  # CHANGE HERE
    
    if selected_aoi not in aoi_options:
        print(f"Error: AOI '{selected_aoi}' not found in aoi_options.")
        print(f"Available options: {list(aoi_options.keys())}")
        return
    
    aoi_filepath = aoi_options[selected_aoi]
    aoi_name = detect_aoi_name(aoi_filepath)
    print(f"Using AOI: {selected_aoi} ({aoi_name})")
    print(f"AOI file path: {aoi_filepath}")
    
    # =============================================================================
    # Step 0: Set working directory and import python libraries
    # =============================================================================
    
    # Projection information
    geog = 'EPSG:4326'  # Geographic projection
    prj = 'EPSG:5070'  # Projected coordinate system
    
    # Working directories
    _current_cwd = Path.cwd().resolve()
    _projdir_path = None
    
    # Running from code dir
    if _current_cwd.name == 'code':
        _parent_dir = _current_cwd.parent
        if (_parent_dir / 'code').exists() and (_parent_dir / 'code').resolve().samefile(_current_cwd.resolve()):
            _projdir_path = _parent_dir
    
    # Running from root dir
    elif (_current_cwd / 'code').is_dir():
        _projdir_path = _current_cwd
    
    # Error handling
    if _projdir_path is None:
        print(f"WARNING: Could not automatically determine project root using standard heuristics from CWD: {_current_cwd}.")
        if _current_cwd.name == 'code':
            _projdir_path = _current_cwd.parent
        else:
            _projdir_path = _current_cwd
    
    projdir = str(_projdir_path.resolve())
    
    # Add the 'code' directory to sys.path
    _custom_functions_path = os.path.join(projdir, 'code')
    if _custom_functions_path not in sys.path:
        sys.path.insert(0, _custom_functions_path)
    
    # Set Current Working Directory to Project Directory
    if Path.cwd().resolve() != Path(projdir).resolve():
        os.chdir(projdir)
        print(f"Project directory set to: {projdir}")
        print(f"Changed working directory to project root: {os.getcwd()}")
    else:
        print(f"Project directory set to: {projdir}")
        print(f"Working directory is already project root: {os.getcwd()}")
    
    # Output directories
    dataAFD = os.path.join(projdir, 'data/output/AFD')
    dataFires = os.path.join(projdir, 'data/output/firePerimeters')
    
    print("Ready!")
    
    # =============================================================================
    # Step 1: Choose an area of interest (AOI)
    # =============================================================================
    
    print(f"\n--- Step 1: Loading Area of Interest ({aoi_name}) ---")
    
    # Load the specified AOI
    fp = os.path.join(projdir, aoi_filepath)
    if not os.path.exists(fp):
        print(f"Error: AOI file not found at {fp}")
        return
    
    aoi = gpd.read_file(fp)
    bounds = aoi.geometry.unary_union.envelope.buffer(10000)
    print(f"AOI '{aoi_name}' loaded successfully")
    print(f"AOI CRS: {aoi.crs}")
    
    # =============================================================================
    # Step 2: Choose a fire perimeter dataset (FIRED v2)
    # =============================================================================
    
    print("\n--- Step 2: Loading Fire Perimeter Dataset ---")
    
    fp = os.path.join(projdir, 'data/input/firePerimeters/FIRED/fired_conus_ak_2000_to_2025_events_merged.gpkg')
    fires = gpd.read_file(fp)
    print(f"Loaded {len(fires)} fire perimeters")
    print(f"Fire perimeter columns: {fires.columns.tolist()}")
    
    # Process fire data
    fires['ig_date'] = pd.to_datetime(fires['ig_date'])
    fires['last_date'] = pd.to_datetime(fires['last_date'])
    fires['tot_ar_km2'] = fires['tot_ar_km2'].astype(float)
    fires = fires[['merge_id', 'member_ids', 'tot_ar_km2', 'ig_year', 'ig_date', 'last_date', 'geometry']]
    fires = fires.drop_duplicates(subset='merge_id', keep='first')
    print(f"After processing: {len(fires)} fire perimeters")
    
    # =============================================================================
    # Step 3a: Filter fire perimeters for AOI
    # =============================================================================
    
    print(f"\n--- Step 3a: Filtering Fires for {aoi_name} ---")
    
    print(f"CRS of 'fires': {fires.crs}")
    print(f"CRS of 'aoi': {aoi.crs}")
    
    if fires.crs != aoi.crs:
        aoi = aoi.to_crs(fires.crs)
    
    aoi_union = aoi.geometry.unary_union
    fires_aoi = fires[fires.geometry.intersects(aoi_union)].copy()
    
    # Detect date range from fire data for consistent naming
    fire_min_year = int(fires_aoi['ig_year'].min())
    fire_max_year = int(fires_aoi['ig_year'].max())
    date_range = f"{fire_min_year}-{fire_max_year}"
    print(f"Fire data spans: {date_range}")
    
    # Export with improved naming scheme: {aoi}_{fire_dataset}_{data_type}_{date_range}_{processing_step}.{ext}
    output_filename = f"{aoi_name}_FIRED_fires_{date_range}_filtered.gpkg"
    fires_aoi.to_file(os.path.join(dataFires, output_filename))
    print(f"Filtered fires to {len(fires_aoi)} records that intersect {aoi_name}.")
    print(f"Exported to: {output_filename}")
    
    # =============================================================================
    # Step 3b: Buffer fire perimeters + take the convex hull
    # =============================================================================
    
    print(f"\n--- Step 3b: Buffering and Convex Hull for {aoi_name} ---")
    
    buffer_dist = 3000
    target_crs_for_output = 'EPSG:5070'
    output_filename = f"{aoi_name}_FIRED_fires_{date_range}_buffered.gpkg"
    
    print(f"Processing {len(fires_aoi)} fire perimeters.")
    print(f"Buffering perimeters by {buffer_dist/1000} km...")
    fires_aoi['geometry'] = fires_aoi.geometry.buffer(buffer_dist)
    print("Calculating convex hull for each buffered perimeter...")
    fires_aoi['geometry'] = fires_aoi.geometry.convex_hull
    
    current_crs = fires_aoi.crs
    if str(current_crs).upper() != target_crs_for_output.upper():
        print(f"Reprojecting 'fires_aoi' from {current_crs} to {target_crs_for_output}...")
        fires_aoi = fires_aoi.to_crs(target_crs_for_output)
    
    output_full_path = os.path.join(dataFires, output_filename)
    fires_aoi.to_file(output_full_path, driver="GPKG")
    print(f"Exported to: {output_full_path}")
    
    # =============================================================================
    # Step 4: Load and process AFD data
    # =============================================================================
    
    print("\n--- Step 4: Loading Active Fire Detection Dataset ---")
    
    final_csv_path = 'data/input/AFD/VIIRS/VIIRS_cat.csv'
    if not os.path.exists(final_csv_path):
        print("Warning: VIIRS_cat.csv not found. Run data cleaning step first.")
        return
    
    afds = pd.read_csv(final_csv_path, dtype={'version': str}).reset_index(drop=True)
    afds = afds.loc[:, ~afds.columns.str.startswith('Unnamed:')]
    print(f"Number of fire detections: {len(afds)}")
    
    # Drop low confidence detections
    N = len(afds)
    afds = afds[afds['confidence'] != 'l']
    print(f"Dropped {N-len(afds)} [{round(((N-len(afds))/N)*100,2)}%] low-confidence obs.")
    
    # =============================================================================
    # Step 5: Create spatial points and subset
    # =============================================================================
    
    print(f"\n--- Step 5: Creating Spatial Points and Subsetting to {aoi_name} ---")
    
    afds['geometry'] = [Point(xy) for xy in zip(afds.longitude, afds.latitude)]
    afds_ll = gpd.GeoDataFrame(afds, geometry='geometry', crs="EPSG:4326")
    afds_ll = afds_ll.to_crs("EPSG:5070")
    afds_ll = afds_ll.reset_index(drop=True)
    afds_ll['afdID'] = afds_ll.index
    
    # Spatial subset
    afds_ll = afds_ll[afds_ll.geometry.within(bounds)]
    print(f"[{len(afds_ll)}({round(len(afds_ll)/len(afds)*100,2)}%)] detections in {aoi_name}.")
    
    # Detect AFD date range for naming
    afds_ll['acq_date'] = pd.to_datetime(afds_ll['acq_date'])
    afd_min_year = int(afds_ll['acq_date'].dt.year.min())
    afd_max_year = int(afds_ll['acq_date'].dt.year.max())
    afd_date_range = f"{afd_min_year}-{afd_max_year}"
    print(f"AFD data spans: {afd_date_range}")
    
    output_filename = f'{aoi_name}_FIRED_viirs_{afd_date_range}_points.gpkg'
    out_fp = os.path.join(dataAFD, output_filename)
    afds_ll.to_file(out_fp)
    print(f"Saved spatial points to: {out_fp}")
    
    # =============================================================================
    # Step 6: Spatial join and duplicate removal
    # =============================================================================
    
    print(f"\n--- Step 6: Spatial Join and Duplicate Removal for {aoi_name} ---")
    
    fires_aoi = fires_aoi.to_crs(afds_ll.crs)
    
    afds_ll_fires = gpd.sjoin(
        afds_ll, fires_aoi[['merge_id', 'tot_ar_km2', 'ig_year', 'ig_date', 'last_date', 'geometry']], 
        how='inner', 
        predicate='within'
    ).drop(columns=['index_right'])
    
    duplicates = afds_ll_fires[afds_ll_fires.duplicated(subset='afdID', keep=False)]
    print(f"Resolving [{len(duplicates)}/{len(afds_ll_fires)}] duplicate obs.")
    
    # Temporal filtering (acq_date already converted to datetime above)
    afds_ll_fires['acq_month'] = afds_ll_fires['acq_date'].dt.month.astype(int)
    afds_ll_fires['acq_year'] = afds_ll_fires['acq_date'].dt.year.astype(int)
    
    afds_ll_fires = afds_ll_fires[
        (afds_ll_fires['acq_date'] >= afds_ll_fires['ig_date'] - timedelta(days=14)) &
        (afds_ll_fires['acq_date'] <= afds_ll_fires['last_date'] + timedelta(days=14))
    ]
    
    # Handle duplicates by keeping largest fire
    duplicates = afds_ll_fires[afds_ll_fires.duplicated(subset='afdID', keep=False)]
    if len(duplicates) > 0:
        print(f"Number of rows before deduplication: {len(afds_ll_fires)}")
        afds_ll_fires = afds_ll_fires.sort_values(['afdID', 'tot_ar_km2'], ascending=[True, False])
        afds_ll_fires = afds_ll_fires.drop_duplicates(subset='afdID', keep='first')
        print(f"Number of rows after deduplication: {len(afds_ll_fires)}")
    
    # =============================================================================
    # Step 7: Process observations and FRP
    # =============================================================================
    
    print(f"\n--- Step 7: Processing Observation Counts and FRP for {aoi_name} ---")
    
    counts = afds_ll_fires.groupby(['merge_id']).size().reset_index(name='count')
    afds_ll_fires = pd.merge(afds_ll_fires, counts, on='merge_id', how='left')
    
    # Calculate pixel area and FRP per area
    afds_ll_fires['pix_area'] = afds_ll_fires['scan'] * afds_ll_fires['track']
    
    # Remove zero FRP observations
    n_zero = afds_ll_fires[afds_ll_fires['frp'] == 0]['frp'].count()
    afds_ll_fires = afds_ll_fires[afds_ll_fires['frp'] > 0]
    print(f"Removed [{n_zero}] observations with FRP == 0")
    
    afds_ll_fires['frp_wkm2'] = afds_ll_fires['frp'] / afds_ll_fires['pix_area']
    
    # Filter to fires with sufficient observations
    n_obs = 10
    afds_fires = afds_ll_fires[afds_ll_fires['count'] >= n_obs]
    print(f"There are {len(afds_fires['merge_id'].unique())} fires with >= {n_obs} obs.")
    
    # Update date range after filtering
    filtered_min_year = int(afds_fires['acq_year'].min())
    filtered_max_year = int(afds_fires['acq_year'].max())
    filtered_date_range = f"{filtered_min_year}-{filtered_max_year}"
    
    output_filename = f'{aoi_name}_FIRED_viirs_{filtered_date_range}_fires.gpkg'
    out_fp = os.path.join(dataAFD, output_filename)
    afds_fires.to_file(out_fp)
    print(f"Saved spatial points to: {out_fp}")
    
    # =============================================================================
    # Step 8: Convert to pixel areas
    # =============================================================================
    
    print(f"\n--- Step 8: Converting AFD to Pixel Areas for {aoi_name} ---")
    
    def pixel_area(point, width, height):
        half_width = width / 2
        half_height = height / 2
        return box(
            point.x - half_width, point.y - half_height,
            point.x + half_width, point.y + half_height
        )
    
    afds_pix = afds_fires.copy()
    afds_pix["geometry"] = afds_pix.apply(
        lambda row: pixel_area(row["geometry"], row["scan"] * 1000, row["track"] * 1000), axis=1
    )
    
    afds_pix = afds_pix.reset_index(drop=True)
    afds_pix['obs_id'] = afds_pix.index
    print(f"Total detections: {len(afds_pix)}")
    
    output_filename = f'{aoi_name}_FIRED_viirs_{filtered_date_range}_pixels.gpkg'
    out_fp = os.path.join(dataAFD, output_filename)
    afds_pix.to_file(out_fp)
    print(f"Saved to {out_fp}")
    
    # =============================================================================
    # Step 9: Create grid and aggregate FRP
    # =============================================================================
    
    print(f"\n--- Step 9: Creating Grid and Aggregating FRP for {aoi_name} ---")
    
    def regular_grid_parallel(extent, res=375, crs_out='EPSG:5070', regions=None):
        """
        Create a regular grid with parallel processing for better performance.
        """
        if not ENABLE_PARALLELIZATION:
            # Fall back to original single-threaded approach
            min_lon, max_lon, min_lat, max_lat = extent
            x_coords = np.arange(min_lon, max_lon, res)
            y_coords = np.arange(min_lat, max_lat, res)
            
            cells = [
                Polygon([(x, y), (x + res, y), (x + res, y + res), (x, y + res)])
                for x in x_coords for y in y_coords
            ]
            
            grid = gpd.GeoDataFrame({'geometry': cells}, crs=crs_out)
            
            if regions is not None:
                if regions.crs != grid.crs:
                    regions = regions.to_crs(grid.crs)
                grid = grid[grid.intersects(regions.unary_union)].copy()
            
            return grid
        
        # Parallel approach
        n_jobs = min(cpu_count(), MAX_CORES) if MAX_CORES else cpu_count()
        
        min_lon, max_lon, min_lat, max_lat = extent
        x_coords = np.arange(min_lon, max_lon, res)
        y_coords = np.arange(min_lat, max_lat, res)
        
        print(f"Creating grid with {len(x_coords)} x {len(y_coords)} = {len(x_coords) * len(y_coords)} cells using {n_jobs} cores...")
        
        # Split x_coords into batches for parallel processing
        batch_size = max(1, len(x_coords) // n_jobs)
        x_batches = [x_coords[i:i + batch_size] for i in range(0, len(x_coords), batch_size)]
        
        # Create arguments for parallel processing
        args_list = [(x_batch, y_coords, res) for x_batch in x_batches]
        
        # Process batches in parallel
        with Pool(n_jobs) as pool:
            cell_batches = pool.map(create_cell_batch, args_list)
        
        # Flatten the results
        cells = [cell for batch in cell_batches for cell in batch]
        
        print(f"Created {len(cells)} grid cells")
        
        grid = gpd.GeoDataFrame({'geometry': cells}, crs=crs_out)
        
        if regions is not None:
            print("Filtering grid to regions...")
            if regions.crs != grid.crs:
                regions = regions.to_crs(grid.crs)
            grid = grid[grid.intersects(regions.unary_union)].copy()
            print(f"Filtered to {len(grid)} grid cells within regions")
        
        return grid
    
    # Get extent
    try:
        coords, extent = get_coords(aoi, buffer=1000, crs='EPSG:5070')
        print(f"Bounding extent for {aoi_name}: {extent}")
    except:
        print("Warning: get_coords function not available, using bounds")
        total_bounds = aoi.total_bounds
        extent = [total_bounds[0] - 1000, total_bounds[2] + 1000, 
                 total_bounds[1] - 1000, total_bounds[3] + 1000]
        print(f"Bounding extent for {aoi_name}: {extent}")
    
    # Create grid
    fires_subset = fires_aoi[fires_aoi['merge_id'].isin(afds_pix['merge_id'].unique())]
    grid = regular_grid_parallel(extent, res=375, crs_out='EPSG:5070', regions=fires_subset)
    
    output_filename = f'{aoi_name}_FIRED_grid_{date_range}_375m.gpkg'
    out_fp = os.path.join(dataFires, output_filename)
    grid.to_file(out_fp, driver="GPKG")
    print(f"Grid saved to: {out_fp}")
    
    # Also save the subset of fires used for grid analysis
    output_filename = f'{aoi_name}_FIRED_fires_{date_range}_subset.gpkg'
    out_fp = os.path.join(dataFires, output_filename)
    fires_subset.to_file(out_fp)
    print(f"Fire subset saved to: {out_fp}")
    
    # =============================================================================
    # Step 10: Sophisticated FRP Grid Aggregation (matching notebook functionality)
    # =============================================================================
    
    print(f"\n--- Step 10: Sophisticated FRP Grid Aggregation for {aoi_name} ---")
    
    # Add grid_index to match notebook functionality
    grid = grid.reset_index(drop=False).rename(columns={'index': 'grid_index'})
    
    # Ensure both datasets are in the same CRS
    if grid.crs != afds_pix.crs:
        afds_pix = afds_pix.to_crs(grid.crs)
    
    # Process fires individually with parallelization (much faster for large datasets)
    if ENABLE_PARALLELIZATION:
        print("Processing fires individually with sophisticated FRP statistics (parallelized)...")
        # Determine optimal number of processes
        n_jobs = min(cpu_count(), MAX_CORES) if MAX_CORES else cpu_count()
        print(f"Using {n_jobs} parallel processes for fire aggregation")
    else:
        print("Processing fires individually with sophisticated FRP statistics (single-threaded)...")
        n_jobs = 1
    
    t0 = time.time()
    
    # Filter to fires with sufficient observations (matching notebook)
    fires_subset = fires_aoi[fires_aoi['merge_id'].isin(afds_pix['merge_id'].unique())]
    
    print(f"Processing {len(fires_subset)} fires with >= {n_obs} detections...")
    
    # Prepare data for parallel processing
    fire_list = []
    for _, fire in fires_subset.iterrows():
        fire_list.append((fire.to_dict(), afds_pix, grid))
    
    # Process fires in parallel with progress tracking
    print("Starting parallel fire processing...")
    
    if n_jobs == 1:
        # Single-threaded fallback with progress bar
        fire_grids = []
        for i, args in enumerate(fire_list):
            if i % 10 == 0:
                print(f"Processing fire {i+1}/{len(fire_list)}: {args[0]['merge_id']}")
            result = process_single_fire_wrapper(args)
            if result is not None:
                fire_grids.append(result)
    else:
        # Multi-threaded processing
        with Pool(n_jobs) as pool:
            # Use pool.map with chunking for better performance
            chunk_size = max(1, len(fire_list) // (n_jobs * 4))
            results = pool.map(process_single_fire_wrapper, fire_list, chunksize=chunk_size)
        
        # Filter out None results
        fire_grids = [result for result in results if result is not None]
    
    print(f"Processed {len(fire_grids)} fires successfully")
    
    # Combine all grids into one
    print("Combining all fire grids...")
    if len(fire_grids) > 0:
        fire_grids_combined = pd.concat(fire_grids, ignore_index=True)
        
        # Remove rows with no detections
        fire_grids_combined = fire_grids_combined[fire_grids_combined['afd_count'] > 0]
        
        t3 = (time.time() - t0) / 60
        print(f"Grid aggregation completed in {t3:.2f} minutes")
        print(f"Grid cells with fire detections: {len(fire_grids_combined)}")
        print(f"Total cumulative FRP: {fire_grids_combined['frp_csum'].sum():.2f} (W/km²)")
        
        # Display top results (matching notebook output)
        print("\nAvailable columns in fire_grids:")
        print(list(fire_grids_combined.columns))
        
        # Define columns to display, checking which ones actually exist
        desired_columns = ['grid_index', 'frp_csum', 'frp_max', 'frp_p90', 'frp_p95', 
                          'frp_p97', 'frp_p99', 'frp_min', 'frp_first', 'afd_count',
                          'day_count', 'night_count', 'unique_days']
        
        available_columns = [col for col in desired_columns if col in fire_grids_combined.columns]
        
        if len(available_columns) > 0:
            top_results = fire_grids_combined.sort_values(by='frp_csum', ascending=False)[available_columns].head(10)
            print(f"\nTop 10 grid cells by cumulative FRP (showing available columns):")
            print(top_results)
        else:
            print("\nUnable to display top results - checking fire_grids structure...")
            print(f"Shape: {fire_grids_combined.shape}")
            print(f"Sample columns: {list(fire_grids_combined.columns)[:10]}")
    else:
        print("No fire grids processed successfully!")
        fire_grids_combined = pd.DataFrame()
    
    # =============================================================================
    # Step 12: Save results
    # =============================================================================
    
    # Save the sophisticated gridstats file
    output_filename = f'{aoi_name}_FIRED_gridstats_{filtered_date_range}_final.gpkg'
    final_output_path = os.path.join(dataAFD, output_filename)
    
    if len(fire_grids_combined) > 0:
        fire_grids_combined.to_file(final_output_path, driver="GPKG")
        print(f"Sophisticated FRP gridstats saved to: {final_output_path}")
    else:
        print(f"No data to save for gridstats file: {final_output_path}")
    
    print(f"\n" + "="*50)
    print(f"FRP Analysis Workflow Complete for {aoi_name}!")
    print(f"="*50)
    print(f"\nKey outputs with improved naming scheme:")
    print(f"  - Fire perimeters: {aoi_name}_FIRED_fires_{date_range}_filtered.gpkg")
    print(f"  - Buffered hulls: {aoi_name}_FIRED_fires_{date_range}_buffered.gpkg")
    print(f"  - AFD points: {aoi_name}_FIRED_viirs_{afd_date_range}_points.gpkg")
    print(f"  - AFD fires: {aoi_name}_FIRED_viirs_{filtered_date_range}_fires.gpkg")
    print(f"  - AFD pixels: {aoi_name}_FIRED_viirs_{filtered_date_range}_pixels.gpkg")
    print(f"  - Grid: {aoi_name}_FIRED_grid_{date_range}_375m.gpkg")
    print(f"  - Fire subset: {aoi_name}_FIRED_fires_{date_range}_subset.gpkg")
    print(f"  - **MAIN OUTPUT** FRP gridstats: {aoi_name}_FIRED_gridstats_{filtered_date_range}_final.gpkg")
    print(f"\nThe gridstats file contains sophisticated FRP statistics including:")
    print(f"  - frp_csum: Cumulative FRP per grid cell")
    print(f"  - frp_p90, frp_p95, frp_p97, frp_p99: FRP percentiles")
    print(f"  - Day/night breakdowns (frp_*_day, frp_*_night)")
    print(f"  - Temporal metrics (first_obs_date, last_obs_date, unique_days)")
    print(f"  - Additional statistics (frp_max, frp_min, afd_count)")


if __name__ == "__main__":
    main() 