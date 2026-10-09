from pathlib import Path
import rasterio

FCF_DIR = Path("Data/FCF")

MAIN_FCFS = [
    "aspect.tif",
    "curvature.tif",
    "distdrainage.tif",
    "distriver.tif",
    "distroads.tif",
    "elevation.tif",
    "ndvi.tif",
    "rfreq_10_201022.tif",
    "slope.tif",
    "twi.tif",
]

print("=" * 70)
print("PHASE 0 — FCF AUDIT")
print("=" * 70)

# Use elevation as the reference grid
ref_path = FCF_DIR / "elevation.tif"

with rasterio.open(ref_path) as ref:
    ref_crs = ref.crs
    ref_width = ref.width
    ref_height = ref.height
    ref_transform = ref.transform
    ref_res = ref.res

for name in MAIN_FCFS:

    path = FCF_DIR / name

    print("\n" + "-" * 70)
    print(name)
    print("-" * 70)

    if not path.exists():
        print("MISSING")
        continue

    with rasterio.open(path) as src:

        tags = src.tags()

        aligned = (
            src.crs == ref_crs
            and src.width == ref_width
            and src.height == ref_height
            and src.transform == ref_transform
            and src.res == ref_res
        )

        print(f"CRS           : {src.crs}")
        print(f"Size          : {src.width} x {src.height}")
        print(f"Resolution    : {src.res}")
        print(f"Data type     : {src.dtypes[0]}")
        print(f"NoData        : {src.nodata}")
        print(f"SCALE_SRC_MIN : {tags.get('SCALE_SRC_MIN')}")
        print(f"SCALE_SRC_MAX : {tags.get('SCALE_SRC_MAX')}")
        print(f"Aligned       : {aligned}")

print("\nAudit complete.")