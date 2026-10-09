from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window, from_bounds
from rasterio.windows import transform as window_transform
from rasterio.windows import bounds as window_bounds
from shapely.geometry import box


# ============================================================
# PATHS
# ============================================================

FCF_DIR = Path("Data/FCF")

REFERENCE = FCF_DIR / "elevation.tif"

RECURRENT = Path(
    "Data/inventory/processed/"
    "recurrent_flood_2010_2014_2022.gpkg"
)

OUT_DIR = Path("Data/inventory/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "recurrent_flood_mask.tif"


# ============================================================
# MAIN 10 FCFs
# ============================================================

FCFS = [
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

BLOCK_SIZE = 1024


print("=" * 80)
print("RASTERIZING RECURRENT FLOOD INVENTORY")
print("=" * 80)


# ============================================================
# READ RECURRENT FLOOD POLYGON
# ============================================================

flood = gpd.read_file(RECURRENT)

with rasterio.open(REFERENCE) as ref:

    target_crs = ref.crs

    print(f"\nReference raster: {REFERENCE.name}")
    print(f"CRS: {target_crs}")
    print(f"Grid: {ref.width:,} x {ref.height:,}")
    print(f"Resolution: {ref.res}")

    if flood.crs != target_crs:
        print("\nReprojecting recurrent flood polygon...")
        flood = flood.to_crs(target_crs)

    # Break MultiPolygon into individual polygons.
    # This makes windowed rasterization much faster.
    parts = flood.explode(
        index_parts=False,
        ignore_index=True
    )

    parts = parts[
        parts.geometry.notna()
        & ~parts.geometry.is_empty
    ].copy()

    print(f"\nFlood polygon parts: {len(parts):,}")

    spatial_index = parts.sindex


    # ========================================================
    # FIND ONLY THE PART OF THE NATIONAL GRID WE NEED
    # ========================================================

    minx, miny, maxx, maxy = parts.total_bounds

    raw_window = from_bounds(
        minx,
        miny,
        maxx,
        maxy,
        transform=ref.transform
    )

    col_start = max(0, int(np.floor(raw_window.col_off)))
    row_start = max(0, int(np.floor(raw_window.row_off)))

    col_end = min(
        ref.width,
        int(np.ceil(raw_window.col_off + raw_window.width))
    )

    row_end = min(
        ref.height,
        int(np.ceil(raw_window.row_off + raw_window.height))
    )

    print("\nProcessing grid window:")
    print(f"Rows:    {row_start:,} to {row_end:,}")
    print(f"Columns: {col_start:,} to {col_end:,}")


    # ========================================================
    # OPEN ALL TEN FCF RASTERS
    # ========================================================

    fcf_sources = {}

    for name in FCFS:
        src = rasterio.open(FCF_DIR / name)

        # Safety check: exact grid alignment
        if (
            src.crs != ref.crs
            or src.width != ref.width
            or src.height != ref.height
            or src.transform != ref.transform
        ):
            raise RuntimeError(
                f"{name} is not aligned with {REFERENCE.name}"
            )

        fcf_sources[name] = src

    print("\n✓ All 10 FCF rasters confirmed aligned.")


    # ========================================================
    # CREATE SPARSE OUTPUT RASTER
    # ========================================================

    profile = ref.profile.copy()

    profile.update(
        dtype="uint8",
        count=1,
        nodata=0,
        compress="DEFLATE",
        tiled=True,
        blockxsize=512,
        blockysize=512,
        BIGTIFF="YES"
    )


    recurrent_pixels = 0
    valid_recurrent_pixels = 0

    missing_by_feature = {
        name: 0 for name in FCFS
    }


    with rasterio.open(
        OUTPUT,
        "w",
        **profile,
        SPARSE_OK="TRUE"
    ) as dst:

        print("\nRasterizing in blocks...")

        for row in range(
            row_start,
            row_end,
            BLOCK_SIZE
        ):

            height = min(
                BLOCK_SIZE,
                row_end - row
            )

            for col in range(
                col_start,
                col_end,
                BLOCK_SIZE
            ):

                width = min(
                    BLOCK_SIZE,
                    col_end - col
                )

                window = Window(
                    col,
                    row,
                    width,
                    height
                )

                # --------------------------------------------
                # Find flood polygons intersecting this block
                # --------------------------------------------

                xmin, ymin, xmax, ymax = window_bounds(
                    window,
                    ref.transform
                )

                window_box = box(
                    xmin,
                    ymin,
                    xmax,
                    ymax
                )

                idx = spatial_index.query(
                    window_box,
                    predicate="intersects"
                )

                if len(idx) == 0:
                    continue

                geometries = [
                    parts.geometry.iloc[i]
                    for i in idx
                ]


                # --------------------------------------------
                # Rasterize recurrent flood polygons
                # --------------------------------------------

                flood_mask = rasterize(
                    [(geom, 1) for geom in geometries],
                    out_shape=(height, width),
                    transform=window_transform(
                        window,
                        ref.transform
                    ),
                    fill=0,
                    all_touched=False,
                    dtype="uint8"
                )

                flood_bool = flood_mask == 1

                if not flood_bool.any():
                    continue

                recurrent_pixels += int(
                    flood_bool.sum()
                )


                # --------------------------------------------
                # Require valid values in ALL 10 FCFs
                # --------------------------------------------

                valid = flood_bool.copy()

                for name, src in fcf_sources.items():

                    data = src.read(
                        1,
                        window=window
                    )

                    nodata = src.nodata

                    if nodata is not None:

                        missing = (
                            flood_bool
                            & (data == nodata)
                        )

                        missing_by_feature[name] += int(
                            missing.sum()
                        )

                        valid &= data != nodata


                valid_recurrent_pixels += int(
                    valid.sum()
                )


                # Write only valid recurrent flood pixels
                output_block = valid.astype(
                    np.uint8
                )

                dst.write(
                    output_block,
                    1,
                    window=window
                )


    # Close FCF datasets
    for src in fcf_sources.values():
        src.close()


# ============================================================
# REPORT
# ============================================================

invalid_pixels = (
    recurrent_pixels
    - valid_recurrent_pixels
)

valid_pct = (
    100 * valid_recurrent_pixels / recurrent_pixels
    if recurrent_pixels > 0
    else 0
)


print("\n" + "=" * 80)
print("RECURRENT FLOOD RASTER CHECK COMPLETE")
print("=" * 80)

print(
    f"Rasterized recurrent pixels: "
    f"{recurrent_pixels:,}"
)

print(
    f"Valid in all 10 FCFs:        "
    f"{valid_recurrent_pixels:,}"
)

print(
    f"Invalid due to FCF NoData:   "
    f"{invalid_pixels:,}"
)

print(
    f"Valid recurrent pixels:      "
    f"{valid_pct:.2f}%"
)

print("\nNoData encountered by FCF:")

for name, count in missing_by_feature.items():
    print(
        f"  {name:<25} {count:,}"
    )

print(
    f"\nSaved aligned flood mask:\n"
    f"{OUTPUT}"
)

print("\nMask meaning:")
print("  1 = recurrent flood + valid in all 10 FCFs")
print("  0 = everything else")