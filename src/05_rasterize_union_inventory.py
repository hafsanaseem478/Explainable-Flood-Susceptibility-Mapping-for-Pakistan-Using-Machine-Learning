from pathlib import Path
from contextlib import ExitStack

import numpy as np
import geopandas as gpd
import rasterio

from rasterio.features import rasterize
from rasterio.windows import (
    Window,
    from_bounds,
    bounds as window_bounds,
    transform as window_transform,
)

from shapely.geometry import box


# ============================================================
# PATHS
# ============================================================

FCF_DIR = Path("Data/FCF")

REFERENCE = FCF_DIR / "elevation.tif"

UNION_VECTOR = Path(
    "Data/inventory/processed/"
    "any_flood_2010_2014_2022.gpkg"
)

OUTPUT = Path(
    "Data/inventory/processed/"
    "historical_flood_union_mask.tif"
)


# ============================================================
# FCFs
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
print("RASTERIZING HISTORICAL FLOOD UNION")
print("=" * 80)


# ============================================================
# READ UNION
# ============================================================

flood = gpd.read_file(
    UNION_VECTOR
)


with rasterio.open(REFERENCE) as ref:

    print(f"\nReference: {REFERENCE.name}")
    print(f"CRS: {ref.crs}")
    print(
        f"Grid: {ref.width:,} x "
        f"{ref.height:,}"
    )
    print(f"Resolution: {ref.res}")


    # ========================================================
    # CRS
    # ========================================================

    if flood.crs != ref.crs:

        print(
            "\nReprojecting union polygon "
            "to FCF CRS..."
        )

        flood = flood.to_crs(
            ref.crs
        )


    # ========================================================
    # EXPLODE MULTIPOLYGON
    # ========================================================

    parts = flood.explode(
        index_parts=False,
        ignore_index=True
    )

    parts = parts[
        parts.geometry.notna()
        & ~parts.geometry.is_empty
    ].copy()

    parts = parts[
        parts.geometry.geom_type.isin(
            ["Polygon", "MultiPolygon"]
        )
    ].copy()


    print(
        f"\nFlood polygon parts: "
        f"{len(parts):,}"
    )


    spatial_index = parts.sindex


    # ========================================================
    # PROCESS ONLY FLOOD BOUNDING WINDOW
    # ========================================================

    minx, miny, maxx, maxy = (
        parts.total_bounds
    )

    raw_window = from_bounds(
        minx,
        miny,
        maxx,
        maxy,
        transform=ref.transform
    )


    col_start = max(
        0,
        int(np.floor(raw_window.col_off))
    )

    row_start = max(
        0,
        int(np.floor(raw_window.row_off))
    )


    col_end = min(
        ref.width,
        int(
            np.ceil(
                raw_window.col_off
                + raw_window.width
            )
        )
    )

    row_end = min(
        ref.height,
        int(
            np.ceil(
                raw_window.row_off
                + raw_window.height
            )
        )
    )


    print("\nProcessing window:")
    print(
        f"Rows:    "
        f"{row_start:,} → {row_end:,}"
    )
    print(
        f"Columns: "
        f"{col_start:,} → {col_end:,}"
    )


    # ========================================================
    # OPEN ALL FCFs ONCE
    # ========================================================

    with ExitStack() as stack:

        sources = {
            name: stack.enter_context(
                rasterio.open(
                    FCF_DIR / name
                )
            )
            for name in FCFS
        }


        # ====================================================
        # ALIGNMENT CHECK
        # ====================================================

        for name, src in sources.items():

            if (
                src.crs != ref.crs
                or src.width != ref.width
                or src.height != ref.height
                or src.transform != ref.transform
            ):

                raise RuntimeError(
                    f"{name} is not aligned "
                    "with elevation.tif"
                )


        print(
            "\n✓ All 10 FCF rasters "
            "confirmed aligned."
        )


        # ====================================================
        # OUTPUT PROFILE
        # ====================================================

        profile = ref.profile.copy()

        profile.update(
            dtype="uint8",
            count=1,
            nodata=0,
            compress="DEFLATE",
            tiled=True,
            blockxsize=512,
            blockysize=512,
            BIGTIFF="YES",
        )


        historical_pixels = 0
        valid_historical_pixels = 0


        missing_by_feature = {
            name: 0
            for name in FCFS
        }


        # ====================================================
        # RASTERIZE BLOCK BY BLOCK
        # ====================================================

        with rasterio.open(
            OUTPUT,
            "w",
            **profile,
            SPARSE_OK="TRUE",
        ) as dst:

            print(
                "\nRasterizing union "
                "in blocks..."
            )


            for row in range(
                row_start,
                row_end,
                BLOCK_SIZE,
            ):

                height = min(
                    BLOCK_SIZE,
                    row_end - row,
                )


                for col in range(
                    col_start,
                    col_end,
                    BLOCK_SIZE,
                ):

                    width = min(
                        BLOCK_SIZE,
                        col_end - col,
                    )


                    window = Window(
                        col,
                        row,
                        width,
                        height,
                    )


                    # ----------------------------------------
                    # Find polygons touching this block
                    # ----------------------------------------

                    xmin, ymin, xmax, ymax = (
                        window_bounds(
                            window,
                            ref.transform,
                        )
                    )


                    window_box = box(
                        xmin,
                        ymin,
                        xmax,
                        ymax,
                    )


                    idx = spatial_index.query(
                        window_box,
                        predicate="intersects",
                    )


                    if len(idx) == 0:
                        continue


                    geometries = [
                        parts.geometry.iloc[i]
                        for i in idx
                    ]


                    # ----------------------------------------
                    # Rasterize
                    # ----------------------------------------

                    flood_mask = rasterize(
                        [
                            (geom, 1)
                            for geom
                            in geometries
                        ],
                        out_shape=(
                            height,
                            width,
                        ),
                        transform=window_transform(
                            window,
                            ref.transform,
                        ),
                        fill=0,
                        all_touched=False,
                        dtype="uint8",
                    )


                    flood_bool = (
                        flood_mask == 1
                    )


                    if not flood_bool.any():
                        continue


                    historical_pixels += int(
                        flood_bool.sum()
                    )


                    # ----------------------------------------
                    # REQUIRE VALID DATA IN ALL 10 FCFs
                    # ----------------------------------------

                    valid = flood_bool.copy()


                    for name, src in sources.items():

                        data = src.read(
                            1,
                            window=window,
                        )

                        nodata = src.nodata


                        if nodata is not None:

                            missing = (
                                flood_bool
                                & (data == nodata)
                            )


                            missing_by_feature[
                                name
                            ] += int(
                                missing.sum()
                            )


                            valid &= (
                                data != nodata
                            )


                    valid_historical_pixels += int(
                        valid.sum()
                    )


                    # ----------------------------------------
                    # WRITE VALID HISTORICAL FLOOD PIXELS
                    # ----------------------------------------

                    output_block = (
                        valid.astype(
                            np.uint8
                        )
                    )


                    dst.write(
                        output_block,
                        1,
                        window=window,
                    )


# ============================================================
# REPORT
# ============================================================

invalid_pixels = (
    historical_pixels
    - valid_historical_pixels
)


valid_pct = (
    100
    * valid_historical_pixels
    / historical_pixels

    if historical_pixels > 0
    else 0
)


print("\n" + "=" * 80)
print("HISTORICAL FLOOD UNION RASTER COMPLETE")
print("=" * 80)


print(
    f"Rasterized historical flood pixels: "
    f"{historical_pixels:,}"
)


print(
    f"Valid in all 10 FCFs:               "
    f"{valid_historical_pixels:,}"
)


print(
    f"Invalid due to FCF NoData:          "
    f"{invalid_pixels:,}"
)


print(
    f"Valid historical pixels:            "
    f"{valid_pct:.2f}%"
)


print("\nNoData encountered by FCF:")

for name, count in missing_by_feature.items():

    print(
        f"  {name:<25}"
        f"{count:,}"
    )


print(
    f"\nSaved:\n"
    f"{OUTPUT}"
)


print("\nMask meaning:")
print(
    "  1 = flooded in at least one "
    "of 2010 / 2014 / 2022"
)
print(
    "      AND valid in all 10 FCFs"
)
print(
    "  0 = outside historical flood union "
    "or invalid FCF pixel"
)