"""
VIIRS Active Fire Detection Downloader

This script downloads VIIRS Active Fire Detection data using NASA's earthaccess API
for fire perimeters within a specified Area of Interest (AOI). It integrates with
the FRP analysis workflow by using the same FIRED fire perimeter preprocessing
steps (filtering, buffering, convex hull) as frp_FIRED_v3.py.

Features:
- Downloads VIIRS/NPP and VIIRS/JPSS1 active fire data (VNP14IMG, VJ114IMG)
- Uses FIRED fire perimeters with AOI filtering and preprocessing
- Configurable AOI selection matching main workflow
- Parallel processing for efficient data extraction
- Progress tracking and granule logging to avoid reprocessing
- Outputs concatenated CSV file compatible with main FRP workflow

Dependencies:
- earthaccess (NASA API)
- Standard geospatial libraries (geopandas, pandas, etc.)

Author: Adapted from Maxwell Cook's original notebook
"""

import sys
import os
import gc
import time
import glob
import traceback
import datetime as dt
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm
from io import BytesIO
import warnings
warnings.simplefilter('ignore')

# NASA earthaccess for VIIRS data
try:
    import earthaccess as ea
except ImportError:
    print("Error: earthaccess not installed. Install with: pip install earthaccess")
    sys.exit(1)

# Import custom functions (with fallback if not available)
try:
    from __functions import get_coords
except ImportError:
    print("Warning: Could not import get_coords from __functions.py")
    print("Using fallback coordinate extraction function")
    
    def get_coords(geom, buffer=None, crs='EPSG:4326'):
        """Fallback function to extract coordinates and extent"""
        # Convert to geographic coordinates for earthaccess
        if geom.crs != crs:
            geom = geom.to_crs(crs)
            
        if buffer:
            geom_buffered = geom.buffer(buffer)
        else:
            geom_buffered = geom
        
        bounds = geom_buffered.total_bounds
        extent = [bounds[0], bounds[2], bounds[1], bounds[3]]  # [minx, maxx, miny, maxy]
        
        # Create coordinate pairs for polygon boundary
        # earthaccess expects [(lon, lat), (lon, lat), ...] format
        coords = []
        for geom_single in geom_buffered.geometry:
            if hasattr(geom_single, 'exterior'):
                exterior_coords = list(geom_single.exterior.coords)
                coords.extend(exterior_coords)
        
        # If no exterior coordinates found, create a bounding box
        if not coords:
            minx, miny, maxx, maxy = bounds
            coords = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)]
        
        print(f"Coordinate extraction: {len(coords)} points, bounds: {bounds}")
        return coords, extent


def detect_aoi_name(aoi_filepath):
    """
    Extract AOI name from the file path for consistent naming.
    
    Parameters:
    aoi_filepath (str): Path to the AOI file
    
    Returns:
    str: AOI name (e.g., 'westUS', 'SRM')
    """
    filename = os.path.basename(aoi_filepath)
    # Remove file extension
    aoi_name = os.path.splitext(filename)[0]
    # Remove common suffixes like projection codes
    aoi_name = aoi_name.replace('_5070', '').replace('_4326', '')
    return aoi_name


class VIIRS_AFD_Downloader:
    """
    Downloads VIIRS Active Fire Data (AFD) for fire perimeters within an AOI
    """
    def __init__(self, start_date, last_date, fire_perimeters,
                 short_names=['VNP14IMG', 'VJ114IMG'],
                 buffer=1000, output_directory=None, 
                 processed_granules=None):
        """
        Args:
            start_date: Initial date for the granule search
            last_date: Final date for the granule search  
            fire_perimeters: GeoDataFrame of fire perimeters to search within
            short_names: VIIRS granule types to download
            buffer: Buffer distance for fire perimeters (meters)
            output_directory: Directory to store downloaded granules
            processed_granules: Set of already processed granule IDs
        """
        # Extract coordinate bounds from fire perimeters
        self.coords, self.extent = get_coords(fire_perimeters, buffer)
        
        # Store parameters
        self.date_range = (str(start_date), str(last_date))
        self.short_names = short_names
        self.output_dir = output_directory
        self.processed_granules = processed_granules or set()
        
        # Load pixel size lookup table
        try:
            lut_path = os.path.join(os.path.dirname(__file__), '../data/tabular/raw/pix_size_lut.csv')
            if os.path.exists(lut_path):
                self.lut = pd.read_csv(lut_path)
            else:
                print("Warning: pix_size_lut.csv not found, creating full VIIRS lookup table")
                # Create comprehensive lookup table for full VIIRS scan (0-6399)
                # VIIRS pixel size varies by scan angle, but for simplicity using nominal 375m
                max_samples = 6400  # Covers observed VIIRS sample range
                self.lut = pd.DataFrame({
                    'sample': range(0, max_samples),
                    'along_scan': [0.375] * max_samples,
                    'along_track': [0.375] * max_samples,
                    'scan_angle': [0.0] * max_samples,
                    'pix_area': [0.375 * 0.375] * max_samples  # km²
                })
        except Exception as e:
            print(f"Warning: Could not load pixel lookup table: {e}")
            # Create minimal fallback covering full VIIRS range
            max_samples = 6400
            self.lut = pd.DataFrame({
                'sample': range(0, max_samples),
                'along_scan': [0.375] * max_samples,
                'along_track': [0.375] * max_samples,
                'scan_angle': [0.0] * max_samples,
                'pix_area': [0.375 * 0.375] * max_samples
            })

    def search_and_open_granules(self):
        """Generate an earthaccess search request and open the granules"""
        print(f"Searching for VIIRS granules from {self.date_range[0]} to {self.date_range[1]}")
        
        # Authenticate with NASA Earthdata
        try:
            auth = ea.login(strategy="interactive")
            print(f"✅ Authentication successful")
            print(f"📋 Authentication info: {type(auth)}")
        except Exception as e:
            print(f"❌ Error authenticating with NASA Earthdata: {e}")
            print("You may need to create an account at https://urs.earthdata.nasa.gov/")
            print("As shown in: https://urs.earthdata.nasa.gov/documentation/for_users/data_access/python")
            return None, None
        
        query = ea.search_data(
            short_name=self.short_names, 
            polygon=self.coords,
            temporal=self.date_range, 
            cloud_hosted=True,
            count=-1
        )
        
        if not query:
            print("No granules found for the specified criteria")
            return None, None
        
        # Get granule identifiers
        granules = [g['umm']['DataGranule']['Identifiers'][0]['Identifier'] for g in query]
        N = len(granules)
        print(f"Found {N} total granules")
        
        if N > 0:
            print(f"Sample granule: {granules[0]}")
            print(f"Search coordinates: {len(self.coords)} coordinate pairs")
            print(f"Search extent: {self.extent}")
        else:
            print("❌ No granules found - this could indicate:")
            print("  - No VIIRS data available for this time/location")
            print("  - Search coordinates may be invalid")
            print("  - Date range might be outside VIIRS operational period")
        
        # Filter out already processed granules
        if self.processed_granules:
            processed = [g.replace('.nc', '') for g in self.processed_granules]
            new_granules = [g for g in granules if g not in processed]
            
            if len(new_granules) == 0:
                print("All granules already processed, skipping...")
                return None, None
            elif len(new_granules) < N:
                print(f"Some granules already processed [{N - len(new_granules)}]")
                query = [item for item in query if item['umm']['DataGranule']['Identifiers'][0]['Identifier'] in new_granules]
                granules = new_granules
            else:
                print(f"Starting processing for [{len(granules)}] new granules")
        
        # Open the fileset
        try:
            fileset = ea.open(query)
            print(f"✅ Fileset opened successfully")
            print(f"📋 Fileset type: {type(fileset)}")
            if len(fileset) > 0:
                print(f"🔍 First file type: {type(fileset[0])}")
                print(f"🔍 First file info: {str(fileset[0])[:100]}...")
            return fileset, granules
        except Exception as e:
            print(f"❌ Error opening granules with earthaccess: {e}")
            print("This could be related to NASA Earthdata authentication")
            print("See: https://urs.earthdata.nasa.gov/documentation/for_users/data_access/python")
            return None, None

    def extract_fire_detections(self, fileset, granules):
        """Extract active fire detections from opened granule fileset"""
        if not fileset or not granules:
            return None
        
        granule_dfs = []
        granule_log_path = os.path.join(self.output_dir, 'logs', 'vnp_vji_processed_granules.txt')
        
        # Ensure log directory exists
        os.makedirs(os.path.dirname(granule_log_path), exist_ok=True)
        
        print(f"Extracting active fires from {len(granules)} granules...")
        
        total_granules_processed = 0
        total_granules_with_fires = 0
        total_fire_pixels = 0
        
        for fp in tqdm(fileset, desc="Processing granules"):
            try:
                total_granules_processed += 1
                df = pd.DataFrame()
                
                # Try different approaches to read the NetCDF file
                swath = None
                
                # Open the granule using the proven working method
                try:
                    swath = xr.open_dataset(fp, phony_dims='access')
                    print(f"  ✅ Successfully opened granule")
                except Exception as e:
                    print(f"  ❌ Failed to open granule: {str(e)[:80]}...")
                    continue
                
                if swath is None:
                    print(f"  ❌ Could not open granule with any method")
                    continue
                
                with swath:
                    granule_id = swath.LocalGranuleID
                    
                    # Check granule metadata
                    print(f"\n--- Processing Granule {total_granules_processed}: {granule_id} ---")
                    print(f"  Platform: {swath.PlatformShortName}")
                    print(f"  Day/Night: {swath.DayNightFlag}")
                    
                    # Check for fire pixels
                    fire_count = swath.FirePix
                    print(f"  FirePix count: {fire_count}")
                    
                    if fire_count == 0:
                        print(f"  ❌ No fire pixels in this granule")
                        continue
                    else:
                        print(f"  ✅ Found {fire_count} fire pixels!")
                        total_granules_with_fires += 1
                        total_fire_pixels += fire_count
                    
                    # Get granule metadata
                    geo_id = swath.VNP03IMG
                    
                    # Check if fire detection arrays exist and have data
                    try:
                        lonfp = swath.variables['FP_longitude'][:]
                        latfp = swath.variables['FP_latitude'][:]
                        print(f"  Fire coordinates: {len(lonfp)} longitude, {len(latfp)} latitude values")
                        
                        if len(lonfp) == 0 or len(latfp) == 0:
                            print(f"  ❌ Empty coordinate arrays despite FirePix > 0")
                            continue
                            
                        # Check coordinate ranges
                        print(f"  Longitude range: {lonfp.min():.3f} to {lonfp.max():.3f}")
                        print(f"  Latitude range: {latfp.min():.3f} to {latfp.max():.3f}")
                        
                    except Exception as e:
                        print(f"  ❌ Error accessing fire coordinates: {e}")
                        continue
                    
                    # Extract fire detection variables
                    frp = swath.variables['FP_power'][:]
                    t4 = swath.variables['FP_T4'][:]
                    t5 = swath.variables['FP_T5'][:]
                    m13 = swath.variables['FP_Rad13'][:]
                    sample = swath.variables['FP_sample'][:]
                    line = swath.variables['FP_line'][:]
                    
                    print(f"  FRP values: min={frp.min():.2f}, max={frp.max():.2f}, mean={frp.mean():.2f}")
                    print(f"  Sample range: {sample.min()} to {sample.max()}")
                    print(f"  Line range: {line.min()} to {line.max()}")
                    
                    # Check if our lookup table covers all sample values
                    sample_values = np.unique(sample)
                    max_sample = sample_values.max()
                    print(f"  Unique samples: {len(sample_values)}, max sample: {max_sample}")
                    print(f"  LUT covers samples up to: {self.lut['sample'].max()}")
                    
                    # Get fire mask
                    try:
                        fire_mask = swath['fire mask'][line, sample].values
                        unique_masks = np.unique(fire_mask)
                        print(f"  Fire mask values: {unique_masks}")
                        
                        # Count confidence levels
                        low_conf = np.sum(fire_mask == 7)
                        nom_conf = np.sum(fire_mask == 8) 
                        high_conf = np.sum(fire_mask == 9)
                        print(f"  Confidence: {high_conf} high, {nom_conf} nominal, {low_conf} low")
                        
                    except Exception as e:
                        print(f"  ❌ Error accessing fire mask: {e}")
                        continue
                
                # Parse timestamp from granule ID
                timestamp = granule_id.split('.')[1:3]
                year = timestamp[0][1:5]
                day = timestamp[0][5:8]
                acqtime = timestamp[1]
                # Use cross-platform date formatting (Windows doesn't support %-m)
                date_obj = dt.datetime.strptime(year+day, '%Y%j')
                acqdate = f"{date_obj.month}/{date_obj.day}/{date_obj.year}"
                
                print(f"  Date: {acqdate}, Time: {acqtime}")
                
                # Build dataframe
                df['longitude'] = lonfp
                df['latitude'] = latfp
                df['j'] = sample  # for pixel size lookup
                df['fire_mask'] = fire_mask
                df['confidence'] = pd.Categorical(df.fire_mask)
                df.confidence = df.confidence.replace(
                    {0:'x', 1:'x', 2:'x', 3:'x', 4:'x', 5:'x', 6:'x', 7:'l', 8:'n', 9:'h'})
                df['frp'] = frp
                df['t4'] = t4
                df['t5'] = t5
                df['m13'] = m13
                df['acq_date'] = acqdate
                df['acq_time'] = acqtime
                df['daynight'] = swath.DayNightFlag
                df['satellite'] = swath.PlatformShortName
                df['short_name'] = swath.ShortName
                df['granule_id'] = granule_id
                df['geo_id'] = geo_id
                
                print(f"  Built dataframe with {len(df)} fire detections")
                
                # Merge with pixel size lookup table
                df = pd.merge(df, self.lut, left_on='j', right_on='sample', how='left')
                df.drop(columns=['j'], inplace=True)
                
                # Check for successful merge
                missing_pix_area = df['pix_area'].isna().sum()
                if missing_pix_area > 0:
                    print(f"  ⚠️  {missing_pix_area} pixels missing pixel area information")
                else:
                    print(f"  ✅ All pixels have pixel area information")
                
                granule_dfs.append(df)
                
                # Log processed granule
                with open(granule_log_path, 'a') as log_file:
                    log_file.write(f"{granule_id}\n")
                
                # Save individual granule CSV
                granules_dir = os.path.join(self.output_dir, "granules")
                os.makedirs(granules_dir, exist_ok=True)
                df.to_csv(os.path.join(granules_dir, f"{granule_id[:-3]}.csv"))
                
            except Exception as e:
                print(f"❌ Error processing granule {total_granules_processed}: {e}")
                print(f"   Error type: {type(e).__name__}")
                # Don't print full traceback unless it's a critical error
                if "h5netcdf" in str(e) or "netcdf" in str(e):
                    print(f"   This appears to be a NetCDF reading issue")
                continue
        
        # Print summary statistics
        print(f"\n📊 EXTRACTION SUMMARY:")
        print(f"  Total granules processed: {total_granules_processed}")
        print(f"  Granules with fire pixels: {total_granules_with_fires}")
        print(f"  Total fire pixels found: {total_fire_pixels}")
        print(f"  Granule DataFrames created: {len(granule_dfs)}")
        
        # Concatenate all granule dataframes
        if granule_dfs:
            print(f"✅ Successfully processed {len(granule_dfs)} granules with fire data")
            fire_data = pd.concat(granule_dfs, ignore_index=True)
            print(f"📋 Combined dataset: {len(fire_data)} total fire detections")
            
            # Show sample of extracted data
            if len(fire_data) > 0:
                print(f"\n🔍 SAMPLE DATA:")
                print(f"  Longitude range: {fire_data['longitude'].min():.3f} to {fire_data['longitude'].max():.3f}")
                print(f"  Latitude range: {fire_data['latitude'].min():.3f} to {fire_data['latitude'].max():.3f}")
                print(f"  FRP range: {fire_data['frp'].min():.2f} to {fire_data['frp'].max():.2f} W")
                print(f"  Confidence levels: {fire_data['confidence'].value_counts().to_dict()}")
                print(f"  Satellites: {fire_data['satellite'].unique()}")
                
            return fire_data
        else:
            print("❌ No fire detections extracted")
            if total_granules_with_fires > 0:
                print(f"⚠️  Found fire pixels in {total_granules_with_fires} granules but failed to extract data")
            return None


def main():
    print("="*60)
    print("VIIRS Active Fire Detection Downloader")
    print("="*60)
    
    # =============================================================================
    # Configuration: Set AOI and date range
    # =============================================================================
    
    # AOI options - must match those in frp_FIRED_v3.py
    aoi_options = {
        'westUS': 'data/input/AOI/westUS_5070.gpkg',
        'srm': 'data/input/AOI/na_cec_eco_l3_srme.gpkg',
        # Add more AOI options here as needed
    }
    
    # Configuration
    selected_aoi = 'srm'  # CHANGE HERE to switch AOIs
    start_date = '2020-05-01'  # CHANGE HERE for date range
    end_date = '2020-05-15'    # CHANGE HERE for date range
    
    if selected_aoi not in aoi_options:
        print(f"Error: AOI '{selected_aoi}' not found in aoi_options.")
        print(f"Available options: {list(aoi_options.keys())}")
        return
    
    aoi_filepath = aoi_options[selected_aoi]
    aoi_name = detect_aoi_name(aoi_filepath)
    
    print(f"Using AOI: {selected_aoi} ({aoi_name})")
    print(f"Date range: {start_date} to {end_date}")
    
    # =============================================================================
    # Set up directories
    # =============================================================================
    
    # Determine project directory
    current_cwd = Path.cwd().resolve()
    
    if current_cwd.name == 'code':
        projdir = current_cwd.parent
    elif (current_cwd / 'code').is_dir():
        projdir = current_cwd
    else:
        projdir = current_cwd
    
    print(f"Project directory: {projdir}")
    
    # Output directories
    dataraw = projdir / 'data' / 'output' / 'AFD' / 'VIIRS'
    dataraw.mkdir(parents=True, exist_ok=True)
    
    # =============================================================================
    # Step 1: Load and filter AOI
    # =============================================================================
    
    print(f"\n--- Step 1: Loading Area of Interest ({aoi_name}) ---")
    
    aoi_full_path = projdir / aoi_filepath
    if not aoi_full_path.exists():
        print(f"Error: AOI file not found at {aoi_full_path}")
        return
    
    aoi = gpd.read_file(aoi_full_path)
    print(f"AOI '{aoi_name}' loaded successfully")
    print(f"AOI CRS: {aoi.crs}")
    
    # =============================================================================
    # Step 2: Load and process FIRED fire perimeters (Steps 3a & 3b from main workflow)
    # =============================================================================
    
    print(f"\n--- Step 2: Loading and Processing FIRED Fire Perimeters ---")
    
    # Load FIRED fire perimeters
    fired_path = projdir / 'data' / 'input' / 'firePerimeters' / 'FIRED' / 'fired_conus_ak_2000_to_2025_events_merged.gpkg'
    if not fired_path.exists():
        print(f"Error: FIRED fire perimeters not found at {fired_path}")
        return
    
    fires = gpd.read_file(fired_path)
    print(f"Loaded {len(fires)} fire perimeters")
    
    # Process fire data
    fires['ig_date'] = pd.to_datetime(fires['ig_date'])
    fires['last_date'] = pd.to_datetime(fires['last_date'])
    fires['tot_ar_km2'] = fires['tot_ar_km2'].astype(float)
    fires = fires[['merge_id', 'tot_ar_km2', 'ig_year', 'ig_date', 'last_date', 'geometry']]
    fires = fires.drop_duplicates(subset='merge_id', keep='first')
    
    # Step 3a: Filter fires for AOI
    print(f"Filtering fires for {aoi_name}...")
    if fires.crs != aoi.crs:
        aoi = aoi.to_crs(fires.crs)
    
    aoi_union = aoi.geometry.unary_union
    fires_aoi = fires[fires.geometry.intersects(aoi_union)].copy()
    print(f"Filtered to {len(fires_aoi)} fires that intersect {aoi_name}")
    
    # Step 3b: Buffer fire perimeters and apply convex hull
    print("Applying 3km buffer and convex hull to fire perimeters...")
    buffer_dist = 3000
    fires_aoi['geometry'] = fires_aoi.geometry.buffer(buffer_dist)
    fires_aoi['geometry'] = fires_aoi.geometry.convex_hull
    
    # Ensure projected CRS for accurate buffering
    target_crs = 'EPSG:5070'
    if str(fires_aoi.crs).upper() != target_crs.upper():
        fires_aoi = fires_aoi.to_crs(target_crs)
    
    print(f"Fire perimeters processed and buffered by {buffer_dist/1000} km")
    
    # =============================================================================
    # Step 3: Filter fires by date range and prepare for download
    # =============================================================================
    
    print(f"\n--- Step 3: Filtering Fires by Date Range ---")
    
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    # Filter fires that overlap with the specified date range
    # Add some buffer to the date range to ensure we don't miss fires
    buffer_days = 14
    start_dt_buffered = start_dt - timedelta(days=buffer_days)
    end_dt_buffered = end_dt + timedelta(days=buffer_days)
    
    fires_filtered = fires_aoi[
        (fires_aoi['ig_date'] <= end_dt_buffered) & 
        (fires_aoi['last_date'] >= start_dt_buffered)
    ].copy()
    
    print(f"Found {len(fires_filtered)} fires active during {start_date} to {end_date} (with ±{buffer_days} day buffer)")
    print(f"Fire date ranges: {fires_filtered['ig_date'].min()} to {fires_filtered['last_date'].max()}")
    
    if len(fires_filtered) == 0:
        print("No fires found in the specified date range!")
        print("Trying without date filtering to see available fires...")
        
        # Show some sample fires for debugging
        sample_fires = fires_aoi.head(10)[['merge_id', 'ig_date', 'last_date', 'tot_ar_km2']]
        print("Sample fires in AOI:")
        print(sample_fires)
        return
    
    # =============================================================================
    # Step 4: Check for already processed granules
    # =============================================================================
    
    print(f"\n--- Step 4: Checking for Previously Downloaded Data ---")
    
    granule_log_path = dataraw / 'logs' / 'vnp_vji_processed_granules.txt'
    if granule_log_path.exists():
        with open(granule_log_path, 'r') as log_file:
            processed_granules = set([line.strip() for line in log_file.readlines()])
    else:
        processed_granules = set()
    
    print(f"Found {len(processed_granules)} already processed granules")
    
    # =============================================================================
    # Step 5: Download VIIRS data for fire perimeters
    # =============================================================================
    
    print(f"\n--- Step 5: Downloading VIIRS Active Fire Data ---")
    
    # Initialize downloader
    downloader = VIIRS_AFD_Downloader(
        start_date=start_date,
        last_date=end_date,
        fire_perimeters=fires_filtered,
        buffer=1000,  # Additional buffer for data search
        output_directory=str(dataraw),
        processed_granules=processed_granules
    )
    
    try:
        # Search and open granules
        start_time = time.time()
        fileset, granules = downloader.search_and_open_granules()
        
        if fileset is None:
            print("No new granules to process")
            return
        
        # Extract fire detections
        print(f"Extracting active fire detections...")
        fire_data = downloader.extract_fire_detections(fileset, granules)
        
        elapsed_time = (time.time() - start_time) / 60
        print(f"Download completed in {elapsed_time:.2f} minutes")
        
        # =============================================================================
        # Step 6: Combine and save final dataset
        # =============================================================================
        
        print(f"\n--- Step 6: Combining and Saving Final Dataset ---")
        
        # Get all individual granule CSV files
        granule_files = list((dataraw / 'granules').glob('*.csv'))
        print(f"Found {len(granule_files)} granule CSV files")
        
        if granule_files:
            # Combine all granule files
            print("Combining all granule files into master dataset...")
            all_data = []
            for file in tqdm(granule_files, desc="Reading granule files"):
                try:
                    df = pd.read_csv(file, index_col=0)
                    all_data.append(df)
                except Exception as e:
                    print(f"Error reading {file}: {e}")
                    continue
            
            if all_data:
                combined_data = pd.concat(all_data, ignore_index=True)
                
                # Save master CSV file compatible with main workflow
                output_filename = f'viirs_snpp_jpss1_afd_{aoi_name}_{start_date}_{end_date}.csv'
                output_path = dataraw / output_filename
                combined_data.to_csv(output_path, index=False)
                
                print(f"Master dataset saved: {output_path}")
                print(f"Total fire detections: {len(combined_data):,}")
                print(f"Date range: {combined_data['acq_date'].min()} to {combined_data['acq_date'].max()}")
                print(f"Satellites: {combined_data['satellite'].unique()}")
                
                # Display summary statistics
                print(f"\nDataset Summary:")
                print(f"- Total detections: {len(combined_data):,}")
                print(f"- High confidence: {len(combined_data[combined_data['confidence'] == 'h']):,}")
                print(f"- Normal confidence: {len(combined_data[combined_data['confidence'] == 'n']):,}")
                print(f"- Low confidence: {len(combined_data[combined_data['confidence'] == 'l']):,}")
                print(f"- Day detections: {len(combined_data[combined_data['daynight'] == 'Day']):,}")
                print(f"- Night detections: {len(combined_data[combined_data['daynight'] == 'Night']):,}")
            
        else:
            print("No granule files found to combine")
            
    except Exception as e:
        print(f"Error during download process: {e}")
        traceback.print_exc()
        return
    
    print("\n" + "="*60)
    print("VIIRS Download Complete!")
    print("="*60)
    print(f"\nOutput files:")
    print(f"- Master dataset: viirs_snpp_jpss1_afd_{aoi_name}_{start_date}_{end_date}.csv")
    print(f"- Individual granules: data/output/AFD/VIIRS/granules/")
    print(f"- Processing log: data/output/AFD/VIIRS/logs/vnp_vji_processed_granules.txt")
    print(f"\nTo use with FRP analysis workflow:")
    print(f"1. Copy/rename the master CSV to: data/input/AFD/VIIRS/VIIRS_cat.csv")
    print(f"2. Run: python code/frp_FIRED_v3.py")


if __name__ == "__main__":
    main()
