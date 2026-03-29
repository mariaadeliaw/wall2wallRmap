import ee
import json
import logging
import os
import sys
from datetime import datetime, timezone

# Force UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

SERVICE_ACCOUNT_FILE = 'auth/wall2wallrmap.json'
GEE_PROJECT = None

# Adapt these to your project layout and year(s)
ASSET_ROOT = 'projects/epistem-490415/assets/RESTORE/'
REGIONS = ['Sumatera', 'Kalimantan', 'JawaBali', 'Sulawesi', 'Nusa', 'Maluku', 'Papua']
YEARS = [2019, 2023]
REGIONS_VECTOR = 'users/hadicu06/IIASA/RESTORE/vector_datasets/classification_regions'

# Command-line options:
#   python mosaic_regions.py --year 2022
#   python mosaic_regions.py            # uses YEARS default


def initialize_ee(service_account_file: str, project: str = None) -> bool:
    if not os.path.exists(service_account_file):
        logger.error(f"Service account file not found: {service_account_file}")
        return False

    try:
        with open(service_account_file, 'r') as f:
            service_account_info = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load service account key: {e}")
        return False

    if service_account_info and not project:
        project = service_account_info.get('project_id')

    try:
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_account_file
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
        logger.info(f"✅ EE initialized via GOOGLE_APPLICATION_CREDENTIALS (project={project})")
        return True
    except Exception as e:
        logger.warning(f"Env credential init failed: {e}")

    try:
        credentials = ee.ServiceAccountCredentials(
            email=service_account_info['client_email'],
            key_file=service_account_file,
        )
        if project:
            ee.Initialize(credentials, project=project)
        else:
            ee.Initialize(credentials)
        logger.info(f"✅ EE initialized via ServiceAccountCredentials (project={project})")
        return True
    except Exception as e:
        logger.warning(f"ServiceAccountCredentials init failed: {e}")

    try:
        ee.Authenticate()
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
        logger.info("✅ EE initialized via interactive auth")
        return True
    except Exception as e:
        logger.error(f"Interactive auth failed: {e}")
        return False


def make_mosaic(year: int,
                asset_root: str = ASSET_ROOT,
                regions=None,
                regions_vector: str = REGIONS_VECTOR):
    if regions is None:
        regions = REGIONS

    region_fc = ee.FeatureCollection(regions_vector)
    region_geom = region_fc.geometry().bounds()  # union bounds for export region

    region_assets = [f"{asset_root}{region}_LULC_{year}_100m" for region in regions]
    logger.info(f"Building mosaic for year={year} from assets:\n  " + "\n  ".join(region_assets))

    source_images = [ee.Image(a) for a in region_assets]
    mosaic = ee.ImageCollection(source_images).mosaic().rename('classification')

    output_asset = f"{asset_root}Indonesia_Mosaic_LULC_{year}_100m"
    logger.info(f"Exporting mosaic to asset: {output_asset}")

    task = ee.batch.Export.image.toAsset(
        image=mosaic,
        description=f"Indonesia_Mosaic_LULC_{year}_100m",
        assetId=output_asset,
        scale=100,
        region=region_geom,
        maxPixels=1e13,
        crs='EPSG:4326'
    )
    task.start()
    logger.info(f"Export task started: {task.id}")
    return output_asset, task.id


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Build and export yearly mosaic from region assets')
    parser.add_argument('--year', type=int, default=None,
                        help='Year to process (e.g. 2021). If omitted, uses YEARS list.')
    args = parser.parse_args()

    processing_years = [args.year] if args.year else YEARS

    if not initialize_ee(SERVICE_ACCOUNT_FILE, project=GEE_PROJECT):
        raise RuntimeError("Earth Engine initialization failed")

    print("\n=== REGION MOSAIC EXPORT SCRIPT ===")
    print(f"Start: {datetime.now(timezone.utc).isoformat()}")
    print(f"Processing years: {processing_years}")

    for year in processing_years:
        try:
            asset_id, task_id = make_mosaic(year)
            logger.info(f"Submission success year={year}: asset={asset_id}, task={task_id}")
        except Exception as e:
            logger.error(f"Year {year} failed: {e}")

    print(f"\nDone: {datetime.now(timezone.utc).isoformat()}")
    print("Check Earth Engine task manager: https://code.earthengine.google.com/tasks")


if __name__ == '__main__':
    main()
