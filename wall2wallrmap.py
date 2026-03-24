import ee
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from collections import defaultdict

# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError)
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# EARTH ENGINE AUTHENTICATION  –  service account or interactive fallback
# ─────────────────────────────────────────────────────────────────────────────
SERVICE_ACCOUNT_FILE = 'auth/wall2wallrmap.json'   # ← set this path
GEE_PROJECT          = None   # None = read project_id from the JSON file itself

def initialize_ee(service_account_file: str, project: str = None) -> bool:
    """
    Initialize Earth Engine with a service account JSON key file.

    Tries two methods in order:
      1. GOOGLE_APPLICATION_CREDENTIALS env-var  →  ee.Initialize(project=...)
      2. Explicit ee.ServiceAccountCredentials   →  ee.Initialize(credentials, ...)

    Falls back to interactive  ee.Authenticate() + ee.Initialize()  if both fail.

    Parameters
    ----------
    service_account_file : str
        Path to the service account JSON key file.
    project : str, optional
        GEE project ID.  If None the project_id field in the JSON is used.

    Returns
    -------
    bool  –  True if EE is ready, False if every method failed.
    """
    service_account_info = None

    # ── validate file ────────────────────────────────────────────────────
    if not os.path.exists(service_account_file):
        logger.error(f"Service account file not found: {service_account_file}")
    else:
        try:
            with open(service_account_file, 'r') as f:
                service_account_info = json.load(f)
        except json.JSONDecodeError as exc:
            logger.error(f"Could not parse service account JSON: {exc}")

    # ── resolve project ──────────────────────────────────────────────────
    if service_account_info and not project:
        project = service_account_info.get('project_id')

    # ── Method 1: GOOGLE_APPLICATION_CREDENTIALS env-var ────────────────
    if service_account_info:
        try:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_account_file
            if project:
                ee.Initialize(project=project)
            else:
                ee.Initialize()
            logger.info(f"✅ EE initialised via env-var credentials (project={project})")
            return True
        except Exception as exc1:
            logger.warning(f"Method 1 failed: {exc1}")

        # ── Method 2: explicit ServiceAccountCredentials ─────────────────
        try:
            credentials = ee.ServiceAccountCredentials(
                email    = service_account_info['client_email'],
                key_file = service_account_file,
            )
            if project:
                ee.Initialize(credentials, project=project)
            else:
                ee.Initialize(credentials)
            logger.info(f"✅ EE initialised via ServiceAccountCredentials (project={project})")
            return True
        except Exception as exc2:
            logger.error(f"Method 2 failed: {exc2}")

    # ── Fallback: interactive authentication ─────────────────────────────
    logger.warning("Service account auth failed – falling back to interactive login.")
    try:
        ee.Authenticate()
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
        logger.info("✅ EE initialised via interactive authentication.")
        return True
    except Exception as exc3:
        logger.error(f"Interactive auth also failed: {exc3}")
        return False


# ── Run initialisation ───────────────────────────────────────────────────────
_ee_ready = initialize_ee(SERVICE_ACCOUNT_FILE, project=GEE_PROJECT)
if not _ee_ready:
    raise RuntimeError(
        "Earth Engine could not be initialised. "
        "Check SERVICE_ACCOUNT_FILE path and project permissions."
    )

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
YEARS               = list(range(2020, 2023))          # 2018 → 2024 inclusive
MAX_POINTS_PER_CLASS = 2000
ASSET_ROOT          = 'projects/epistem-490415/assets/RESTORE/'
# DRIVE_FOLDER      = 'RESTORE_LULC_2025'              # uncomment if needed
REGIONS             = ['Sumatera', 'Kalimantan', 'JawaBali', 'Sulawesi',
                        'Nusa', 'Maluku', 'Papua']

CLASS_CONFIG = [
    {'id':  1, 'name': 'Undisturbed dry-land forest',  'color': '#006400'},
    {'id':  2, 'name': 'Logged-over dry-land forest',  'color': '#228B22'},
    {'id':  3, 'name': 'Undisturbed mangrove',          'color': '#4169E1'},
    {'id':  4, 'name': 'Logged-over mangrove',          'color': '#87CEEB'},
    {'id':  5, 'name': 'Undisturbed swamp forest',      'color': '#2E8B57'},
    {'id':  6, 'name': 'Logged-over swamp forest',      'color': '#8FBC8F'},
    {'id':  7, 'name': 'Agroforestry',                  'color': '#9ACD32'},
    {'id':  8, 'name': 'Plantation forest',             'color': '#32CD32'},
    {'id':  9, 'name': 'Rubber monoculture',            'color': '#8B4513'},
    {'id': 10, 'name': 'Oil palm monoculture',          'color': '#FF8C00'},
    {'id': 11, 'name': 'Other monoculture',             'color': '#DAA520'},
    {'id': 12, 'name': 'Grass/savanna',                 'color': '#ADFF2F'},
    {'id': 13, 'name': 'Shrub',                         'color': '#90EE90'},
    {'id': 14, 'name': 'Cropland',                      'color': '#FFFF00'},
    {'id': 15, 'name': 'Settlement',                    'color': '#FF0000'},
    {'id': 16, 'name': 'Cleared land',                  'color': '#D2B48C'},
    {'id': 17, 'name': 'Waterbody',                     'color': '#0000FF'},
]

CLASS_IDS    = [c['id']   for c in CLASS_CONFIG]
CLASS_NAMES  = {c['id']: c['name']  for c in CLASS_CONFIG}
CLASS_COLORS = [c['color'] for c in CLASS_CONFIG]

# ─────────────────────────────────────────────────────────────────────────────
# 2. STATUS TRACKER
# ─────────────────────────────────────────────────────────────────────────────
class StatusTracker:
    """Lightweight tracker for per-region/per-year processing status."""

    # Step indices (order matters for display)
    STEPS = [
        'composite_loaded',
        'points_filtered',
        'regions_sampled',
        'classifier_trained',
        'classification_done',
        'export_started',
    ]
    STEP_LABELS = {
        'composite_loaded':    'Annual composite loaded',
        'points_filtered':     'Training points filtered',
        'regions_sampled':     'Regions sampled',
        'classifier_trained':  'Classifier trained (RF-50)',
        'classification_done': 'Classification complete',
        'export_started':      'Export task submitted',
    }

    def __init__(self):
        # results[year][region] = dict with keys: status, steps, task_id,
        #                          n_orig_points, error, t_start, t_end
        self.results   = defaultdict(dict)
        self.run_start = datetime.now(timezone.utc)

    # ── record helpers ─────────────────────────────────────────────────────
    def start(self, year, region):
        self.results[year][region] = {
            'status':        'running',
            'steps':         {s: False for s in self.STEPS},
            'task_id':        None,
            'n_orig_points':  None,
            'error':          None,
            't_start':        time.time(),
            't_end':          None,
        }

    def mark_step(self, year, region, step):
        self.results[year][region]['steps'][step] = True

    def set_meta(self, year, region, **kwargs):
        self.results[year][region].update(kwargs)

    def finish_ok(self, year, region):
        r = self.results[year][region]
        r['status'] = 'success'
        r['t_end']  = time.time()

    def finish_err(self, year, region, error):
        r = self.results[year][region]
        r['status'] = 'failed'
        r['error']  = str(error)
        r['t_end']  = time.time()

    # ── display helpers ────────────────────────────────────────────────────
    @staticmethod
    def _elapsed(t_start, t_end):
        secs = int((t_end or time.time()) - t_start)
        m, s = divmod(secs, 60)
        return f"{m}m {s:02d}s"

    def print_region_summary(self, year, region):
        r   = self.results[year][region]
        ok  = r['status'] == 'success'
        ico = '[OK]' if ok else '[FAIL]'
        elapsed = self._elapsed(r['t_start'], r['t_end'])
        print(f"\n  {ico} {region} ({year})  [{elapsed}]")

        for step in self.STEPS:
            done   = r['steps'][step]
            bullet = '  [v]' if done else '  [x]'
            print(f"    {bullet}  {self.STEP_LABELS[step]}")

        if r['n_orig_points'] is not None:
            print(f"       (i) Original points in region: {r['n_orig_points']}")
        if r['task_id']:
            print(f"       ID  Task ID: {r['task_id']}")
        if r['error']:
            print(f"       (!) Error: {r['error']}")

    def print_full_summary(self):
        run_end = datetime.now(timezone.utc)
        total_s = int((run_end - self.run_start).total_seconds())
        total_m, total_s2 = divmod(total_s, 60)

        print("\n" + "="*70)
        print("  PROCESSING SUMMARY")
        print(f"  Run started : {self.run_start.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"  Run ended   : {run_end.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"  Total wall-clock time: {total_m}m {total_s2:02d}s")
        print("="*70)

        all_success = 0
        all_failed  = 0

        # ── per-year table ─────────────────────────────────────────────────
        for year in sorted(self.results.keys()):
            yr_results  = self.results[year]
            n_ok  = sum(1 for r in yr_results.values() if r['status'] == 'success')
            n_err = sum(1 for r in yr_results.values() if r['status'] == 'failed')
            all_success += n_ok
            all_failed  += n_err

            print(f"\n  +-- {year}  (success: {n_ok}/{len(yr_results)}, "
                  f"failed: {n_err}/{len(yr_results)}) --")

            for region in REGIONS:
                if region not in yr_results:
                    print(f"  |   [--] {region:<14}  skipped / not run")
                    continue
                r       = yr_results[region]
                ok      = r['status'] == 'success'
                ico     = '[OK]  ' if ok else '[FAIL]'
                elapsed = self._elapsed(r['t_start'], r['t_end'])
                pts     = f"pts={r['n_orig_points']}" if r['n_orig_points'] else ''
                tid     = f"task={r['task_id']}"      if r['task_id']       else ''
                err_str = f"  <- {r['error'][:60]}"   if r['error']         else ''
                meta    = '  '.join(filter(None, [pts, tid]))
                print(f"  |   {ico}  {region:<14}  {elapsed:<9}  {meta}{err_str}")

            print(f"  +" + "-"*60)

        # ── overall ───────────────────────────────────────────────────────
        total = all_success + all_failed
        pct   = int(100 * all_success / total) if total else 0
        print(f"\n  OVERALL: {all_success}/{total} tasks succeeded ({pct}%)")

        if all_failed:
            print("\n  (!) FAILED TASKS:")
            for year in sorted(self.results.keys()):
                for region, r in self.results[year].items():
                    if r['status'] == 'failed':
                        print(f"     * {year} / {region}: {r.get('error','unknown error')}")

        print("\n  Assets location  : " + ASSET_ROOT)
        print("  Tasks monitor    : https://code.earthengine.google.com/tasks")
        print("="*70 + "\n")


# Global tracker instance
tracker = StatusTracker()


# ─────────────────────────────────────────────────────────────────────────────
# 3. HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def get_annual_composite(region_geom, year):
    """
    Load USGS Annual Composite for a given region and year,
    then append NDVI, NDWI, NDBI bands.
    """
    annual_col = (ee.ImageCollection("LANDSAT/COMPOSITES/C02/T1_L2_ANNUAL")
                    .filterDate(f'{year}-01-01', f'{year}-12-31'))
    annual_img = annual_col.first()
    composite  = annual_img.select(
        ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'thermal'],
        ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'temp']
    ).clip(region_geom)

    ndvi = composite.normalizedDifference(['nir', 'red']).rename('ndvi')
    ndwi = composite.normalizedDifference(['green', 'nir']).rename('ndwi')
    ndbi = composite.normalizedDifference(['swir1', 'nir']).rename('ndbi')

    return composite.addBands([ndvi, ndwi, ndbi])


def sample_training_data(fc, max_per_class):
    """Down-sample training points to at most `max_per_class` per class."""
    classes = ee.List(CLASS_IDS)

    def sample_class(c):
        c           = ee.Number(c)
        class_pts   = fc.filter(ee.Filter.eq('kelas', c))
        return class_pts.randomColumn('rand').limit(max_per_class)

    sampled_list = classes.map(sample_class)
    return ee.FeatureCollection(sampled_list).flatten()


# ─────────────────────────────────────────────────────────────────────────────
# 4. REGION PROCESSOR
# ─────────────────────────────────────────────────────────────────────────────
def process_region(region_name, year, training_all, regions_fc):
    """
    Full pipeline for one (region, year) pair:
      composite → sample → train RF → classify → export to Asset.

    Updates the global `tracker` at each step.
    Returns the classified ee.Image (or raises on error).
    """
    tracker.start(year, region_name)

    # ── geometry ──────────────────────────────────────────────────────────
    region_geom = (regions_fc
                   .filter(ee.Filter.eq('region_name', region_name))
                   .geometry())

    # STEP 1 – composite
    composite = get_annual_composite(region_geom, year)
    tracker.mark_step(year, region_name, 'composite_loaded')

    # STEP 2 – training points
    region_points  = (training_all
                      .filterBounds(region_geom)
                      .distinct('.geo'))
    orig_count     = region_points.size().getInfo()
    tracker.set_meta(year, region_name, n_orig_points=orig_count)

    # Special handling: halve points for Sumatera to avoid memory overflow
    if region_name == 'Sumatera':
        region_points = region_points.randomColumn().filter('random < 0.5')
        print(f"    [i] Sumatera: reduced to ~50% of {orig_count} points")

    tracker.mark_step(year, region_name, 'points_filtered')

    # STEP 3 – sampleRegions
    sampled_points = composite.sampleRegions(
        collection=region_points,
        properties=['kelas'],
        scale=30,
        tileScale=16,
        geometries=True
    )
    # Remove points with any null band value
    band_names     = composite.bandNames()
    sampled_points = sampled_points.filter(ee.Filter.notNull(band_names))
    tracker.mark_step(year, region_name, 'regions_sampled')

    # STEP 4 – per-class sampling + train RF
    sampled_training = sample_training_data(sampled_points, MAX_POINTS_PER_CLASS)

    classifier = (ee.Classifier.smileRandomForest(50)
                    .train(
                        features       = sampled_training,
                        classProperty  = 'kelas',
                        inputProperties= composite.bandNames()
                    ))
    tracker.mark_step(year, region_name, 'classifier_trained')

    # STEP 5 – classify
    classified = composite.classify(classifier).rename('classification')
    tracker.mark_step(year, region_name, 'classification_done')

    # STEP 6 – export to Asset
    asset_id = f"{ASSET_ROOT}{region_name}_LULC_{year}_100m"
    task     = ee.batch.Export.image.toAsset(
        image       = classified,
        description = f"{region_name}_LULC_{year}_100m",
        assetId     = asset_id,
        scale       = 100,
        region      = region_geom,
        maxPixels   = 1e13,
        crs         = 'EPSG:4326'
    )
    task.start()
    tracker.set_meta(year, region_name, task_id=task.id)
    tracker.mark_step(year, region_name, 'export_started')
    tracker.finish_ok(year, region_name)

    # (Optional) Drive export – uncomment if needed:
    # drive_task = ee.batch.Export.image.toDrive(
    #     image           = classified,
    #     description     = f"{region_name}_LULC_{year}_100m",
    #     folder          = 'RESTORE_LULC_2025',
    #     fileNamePrefix  = f"{region_name}_LULC_{year}_100m",
    #     scale           = 100,
    #     region          = region_geom,
    #     maxPixels       = 1e13,
    #     fileFormat      = 'GeoTIFF'
    # )
    # drive_task.start()

    return classified


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN LOOP  –  years × regions
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── load shared assets once ──────────────────────────────────────────
    print("Loading shared input data...")
    training_all = ee.FeatureCollection(
        'projects/ee-rg2icraf/assets/Indonesia_lulc_Sample')
    total_pts    = training_all.size().getInfo()
    print(f"  [OK] Training data loaded. Total points: {total_pts}")

    regions_fc   = ee.FeatureCollection(
        "users/hadicu06/IIASA/RESTORE/vector_datasets/classification_regions")
    print("  [OK] Region boundaries loaded.\n")

    # ── outer loop: years ────────────────────────────────────────────────
    total_tasks   = len(YEARS) * len(REGIONS)
    current_task  = 0

    print("=" * 70)
    print(f"  STARTING MULTI-YEAR CLASSIFICATION  ({YEARS[0]}-{YEARS[-1]})")
    print(f"     Regions : {', '.join(REGIONS)}")
    print(f"     Total tasks : {total_tasks}")
    print("=" * 70)

    for year in YEARS:
        print(f"\n{'-'*70}")
        print(f"  YEAR {year}")
        print(f"{'-'*70}")

        year_success = 0
        year_failed  = 0

        for region in REGIONS:
            current_task += 1
            progress = f"[{current_task:>3}/{total_tasks}]"
            print(f"\n{progress}  > {region}  ({year})")

            try:
                process_region(region, year, training_all, regions_fc)
                tracker.print_region_summary(year, region)
                year_success += 1
                print(f"  [OK] {region} ({year}) -- export task submitted")

            except Exception as exc:
                tracker.finish_err(year, region, exc)
                tracker.print_region_summary(year, region)
                year_failed += 1
                print(f"  [FAIL] {region} ({year}) -- FAILED: {exc}")

        # ── per-year mini-summary ────────────────────────────────────────
        print(f"\n  Year {year} complete: "
              f"{year_success} succeeded, {year_failed} failed")

    # ── final summary ────────────────────────────────────────────────────
    tracker.print_full_summary()


if __name__ == '__main__':
    main()

