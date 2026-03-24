import ee

# Initialize Earth Engine
ee.Initialize()

# ---------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------
YEAR = 2024
MAX_POINTS_PER_CLASS = 2000
ASSET_ROOT = 'projects/ee-gautamdadhich3392/assets/Restore2024/'
# DRIVE_FOLDER = 'RESTORE_LULC_2025'   # optional, uncomment if needed
REGIONS = ['Sumatera', 'Kalimantan', 'JawaBali', 'Sulawesi', 'Nusa', 'Maluku', 'Papua']

# Class configuration (from your legend)
CLASS_CONFIG = [
    {'id': 1, 'name': 'Undisturbed dry-land forest', 'color': '#006400'},
    {'id': 2, 'name': 'Logged-over dry-land forest', 'color': '#228B22'},
    {'id': 3, 'name': 'Undisturbed mangrove', 'color': '#4169E1'},
    {'id': 4, 'name': 'Logged-over mangrove', 'color': '#87CEEB'},
    {'id': 5, 'name': 'Undisturbed swamp forest', 'color': '#2E8B57'},
    {'id': 6, 'name': 'Logged-over swamp forest', 'color': '#8FBC8F'},
    {'id': 7, 'name': 'Agroforestry', 'color': '#9ACD32'},
    {'id': 8, 'name': 'Plantation forest', 'color': '#32CD32'},
    {'id': 9, 'name': 'Rubber monoculture', 'color': '#8B4513'},
    {'id': 10, 'name': 'Oil palm monoculture', 'color': '#FF8C00'},
    {'id': 11, 'name': 'Other monoculture', 'color': '#DAA520'},
    {'id': 12, 'name': 'Grass/savanna', 'color': '#ADFF2F'},
    {'id': 13, 'name': 'Shrub', 'color': '#90EE90'},
    {'id': 14, 'name': 'Cropland', 'color': '#FFFF00'},
    {'id': 15, 'name': 'Settlement', 'color': '#FF0000'},
    {'id': 16, 'name': 'Cleared land', 'color': '#D2B48C'},
    {'id': 17, 'name': 'Waterbody', 'color': '#0000FF'}
]

CLASS_IDS = [c['id'] for c in CLASS_CONFIG]
CLASS_NAMES = {c['id']: c['name'] for c in CLASS_CONFIG}
CLASS_COLORS = [c['color'] for c in CLASS_CONFIG]

# ---------------------------------------------------------------------
# 2. LOAD INPUT DATA
# ---------------------------------------------------------------------
print("📡 Loading input data...")

training_all = ee.FeatureCollection('projects/ee-rg2icraf/assets/Indonesia_lulc_Sample')
print(f"✅ Training data loaded. Total points: {training_all.size().getInfo()}")

regions_fc = ee.FeatureCollection("users/hadicu06/IIASA/RESTORE/vector_datasets/classification_regions")
print("✅ Region boundaries loaded.")

# ---------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# ---------------------------------------------------------------------
def get_annual_composite(region, year):
    """Load USGS Annual Composite for a given region and year."""
    annual_col = ee.ImageCollection("LANDSAT/COMPOSITES/C02/T1_L2_ANNUAL") \
        .filterDate(f'{year}-01-01', f'{year}-12-31')
    annual_img = annual_col.first()
    composite = annual_img.select(
        ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'thermal'],
        ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'temp']
    ).clip(region)

    ndvi = composite.normalizedDifference(['nir', 'red']).rename('ndvi')
    ndwi = composite.normalizedDifference(['green', 'nir']).rename('ndwi')
    ndbi = composite.normalizedDifference(['swir1', 'nir']).rename('ndbi')

    return composite.addBands([ndvi, ndwi, ndbi])

def sample_training_data(fc, max_per_class):
    """Sample training data to max points per class."""
    classes = ee.List(CLASS_IDS)
    def sample_class(c):
        c = ee.Number(c)
        class_points = fc.filter(ee.Filter.eq('kelas', c))
        return class_points.randomColumn('rand').limit(max_per_class)
    sampled_list = classes.map(sample_class)
    return ee.FeatureCollection(sampled_list).flatten()

def process_region(region_name):
    """Process one region: classify and start asset export."""
    print("\n" + "-"*60)
    print(f"🔍 Processing region: {region_name}")

    region_geom = regions_fc.filter(ee.Filter.eq('region_name', region_name)).geometry()

    # 1. Get annual composite
    composite = get_annual_composite(region_geom, YEAR)
    print("  ✅ Annual composite loaded")

    # 2. Get training points in region (distinct coordinates)
    region_points = training_all.filterBounds(region_geom).distinct('.geo')
    orig_count = region_points.size().getInfo()
    print(f"  📊 Original points in region: {orig_count}")

    # ---- SPECIAL HANDLING FOR SUMATERA ----
    if region_name == 'Sumatera':
        # Reduce points by ~50% to avoid memory overflow during sampleRegions
        region_points = region_points.randomColumn().filter('random < 0.5')
        print("  🔽 Reduced Sumatra points by 50% for sampling")

    # 3. Sample composite at points
    sampled_points = composite.sampleRegions(
        collection=region_points,
        properties=['kelas'],
        scale=30,
        tileScale=16,
        geometries=True
    )

    # 4. Remove points with null band values
    band_names = composite.bandNames()
    sampled_points = sampled_points.filter(ee.Filter.notNull(band_names))

    # 5. Sample per class
    sampled_training = sample_training_data(sampled_points, MAX_POINTS_PER_CLASS)
    sampled_size = sampled_training.limit(1).size().getInfo()
    print(f"  📊 Sampled points after filtering & per‑class limit: ~{sampled_size}+")

    # 6. Train classifier
    classifier = ee.Classifier.smileRandomForest(50).train(
        features=sampled_training,
        classProperty='kelas',
        inputProperties=composite.bandNames()
    )
    print("  ✅ Classifier trained")

    # 7. Classify
    classified = composite.classify(classifier).rename('classification')
    print("  ✅ Classification complete")

    # 8. Start export task (to asset)
    asset_id = ASSET_ROOT + region_name + '_LULC_' + str(YEAR) + '_100m'
    task = ee.batch.Export.image.toAsset(
        image=classified,
        description=region_name + '_LULC_' + str(YEAR) + '_100m',
        assetId=asset_id,
        scale=100,
        region=region_geom,
        maxPixels=1e13,
        crs='EPSG:4326'
    )
    task.start()
    print(f"  📤 Export task started: {task.id}")

    # (Optional) also start a Drive export – uncomment if needed
    # drive_task = ee.batch.Export.image.toDrive(
    #     image=classified,
    #     description=region_name + '_LULC_' + str(YEAR) + '_100m',
    #     folder='RESTORE_LULC_2025',
    #     fileNamePrefix=region_name + '_LULC_' + str(YEAR) + '_100m',
    #     scale=100,
    #     region=region_geom,
    #     maxPixels=1e13,
    #     fileFormat='GeoTIFF'
    # )
    # drive_task.start()
    # print(f"  📤 Drive export started: {drive_task.id}")

    return classified

# ---------------------------------------------------------------------
# 4. RUN FOR ALL REGIONS
# ---------------------------------------------------------------------
print("\n" + "="*60)
print("🚀 STARTING REGION‑WISE CLASSIFICATION & EXPORT")
print("="*60)

region_images = []
for i, region in enumerate(REGIONS, 1):
    try:
        img = process_region(region)
        region_images.append(img)
        print(f"✅ Completed: {region} ({i}/{len(REGIONS)})")
    except Exception as e:
        print(f"❌ Failed: {region} - {e}")

print("\n" + "="*60)
print("✅ ALL TASKS SUBMITTED!")
print("="*60)
print("Check the Earth Engine Tasks tab for progress:")
print("https://code.earthengine.google.com/tasks")
print(f"\n📌 Exported assets will appear in: {ASSET_ROOT}")
print("   (e.g., Sumatera_LULC_2024_100m, Kalimantan_LULC_2024_100m, ...)")
