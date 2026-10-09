from pathlib import Path
from contextlib import ExitStack

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import Window, from_bounds
from shapely import contains_xy, prepare
from shapely.geometry import Point


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42

N_FLOOD = 10_000
N_NONFLOOD = 10_000

BATCH_SIZE = 30_000
BLOCK_SIZE = 1024


# ============================================================
# PATHS
# ============================================================

FCF_DIR = Path("Data/FCF")

FLOOD_MASK = Path(
    "Data/inventory/processed/recurrent_flood_mask.tif"
)

RECURRENT_VECTOR = Path(
    "Data/inventory/processed/"
    "recurrent_flood_2010_2014_2022.gpkg"
)

ANY_FLOOD = Path(
    "Data/inventory/processed/"
    "any_flood_2010_2014_2022.gpkg"
)

OUT_DIR = Path("Data/samples")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_OUTPUT = OUT_DIR / "training_samples_raw.csv"
GPKG_OUTPUT = OUT_DIR / "training_samples.gpkg"

REFERENCE = FCF_DIR / "elevation.tif"


FCFS = {
    "aspect": "aspect.tif",
    "curvature": "curvature.tif",
    "distdrainage": "distdrainage.tif",
    "distriver": "distriver.tif",
    "distroads": "distroads.tif",
    "elevation": "elevation.tif",
    "ndvi": "ndvi.tif",
    "rainfall_frequency": "rfreq_10_201022.tif",
    "slope": "slope.tif",
    "twi": "twi.tif",
}


rng = np.random.default_rng(RANDOM_SEED)


print("=" * 80)
print("BUILDING BALANCED TRAINING SAMPLE")
print("=" * 80)


# ============================================================
# HELPER: READ VALUES FROM SPECIFIC PIXELS
# ============================================================

def read_pixels(src, rows, cols):

    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)

    values = np.empty(
        len(rows),
        dtype=src.dtypes[0]
    )

    block_h, block_w = src.block_shapes[0]

    block_rows = rows // block_h
    block_cols = cols // block_w

    groups = {}

    for i, key in enumerate(
        zip(block_rows, block_cols)
    ):
        groups.setdefault(key, []).append(i)

    for (br, bc), indices in groups.items():

        indices = np.asarray(
            indices,
            dtype=np.int64
        )

        row0 = br * block_h
        col0 = bc * block_w

        height = min(
            block_h,
            src.height - row0
        )

        width = min(
            block_w,
            src.width - col0
        )

        window = Window(
            col0,
            row0,
            width,
            height
        )

        block = src.read(
            1,
            window=window
        )

        local_rows = rows[indices] - row0
        local_cols = cols[indices] - col0

        values[indices] = block[
            local_rows,
            local_cols
        ]

    return values


# ============================================================
# OPEN ALL FCF RASTERS ONCE
# ============================================================

with ExitStack() as stack:

    sources = {
        feature: stack.enter_context(
            rasterio.open(FCF_DIR / filename)
        )
        for feature, filename in FCFS.items()
    }

    ref = sources["elevation"]

    print("\nReference grid:")
    print(f"  CRS: {ref.crs}")
    print(
        f"  Size: "
        f"{ref.width:,} x {ref.height:,}"
    )
    print(f"  Resolution: {ref.res}")


    # ========================================================
    # VERIFY ALL 10 RASTERS ARE EXACTLY ALIGNED
    # ========================================================

    print("\nChecking FCF alignment...")

    for feature, src in sources.items():

        if (
            src.crs != ref.crs
            or src.width != ref.width
            or src.height != ref.height
            or src.transform != ref.transform
        ):
            raise RuntimeError(
                f"{feature} is not aligned "
                "with elevation.tif"
            )

    print("✓ All 10 FCF rasters are aligned.")


    # ========================================================
    # 1. SAMPLE 10,000 RECURRENT FLOOD PIXELS
    # ========================================================

    print("\n[1/4] Sampling recurrent flood pixels...")

    recurrent = gpd.read_file(
        RECURRENT_VECTOR
    )

    with rasterio.open(FLOOD_MASK) as mask:

        # Extra safety check
        if (
            mask.crs != ref.crs
            or mask.width != ref.width
            or mask.height != ref.height
            or mask.transform != ref.transform
        ):
            raise RuntimeError(
                "recurrent_flood_mask.tif is not "
                "aligned with the FCF grid."
            )

        if recurrent.crs != mask.crs:
            recurrent = recurrent.to_crs(
                mask.crs
            )

        minx, miny, maxx, maxy = (
            recurrent.total_bounds
        )

        raw_window = from_bounds(
            minx,
            miny,
            maxx,
            maxy,
            transform=mask.transform
        )

        row_start = max(
            0,
            int(np.floor(raw_window.row_off))
        )

        col_start = max(
            0,
            int(np.floor(raw_window.col_off))
        )

        row_end = min(
            mask.height,
            int(
                np.ceil(
                    raw_window.row_off
                    + raw_window.height
                )
            )
        )

        col_end = min(
            mask.width,
            int(
                np.ceil(
                    raw_window.col_off
                    + raw_window.width
                )
            )
        )

        flood_indices = []

        for row0 in range(
            row_start,
            row_end,
            BLOCK_SIZE
        ):

            height = min(
                BLOCK_SIZE,
                row_end - row0
            )

            for col0 in range(
                col_start,
                col_end,
                BLOCK_SIZE
            ):

                width = min(
                    BLOCK_SIZE,
                    col_end - col0
                )

                window = Window(
                    col0,
                    row0,
                    width,
                    height
                )

                block = mask.read(
                    1,
                    window=window
                )

                rr, cc = np.where(
                    block == 1
                )

                if len(rr) == 0:
                    continue

                global_rows = rr + row0
                global_cols = cc + col0

                flat = (
                    global_rows.astype(np.int64)
                    * mask.width
                    + global_cols.astype(np.int64)
                )

                flood_indices.append(flat)


        flood_indices = np.concatenate(
            flood_indices
        )

        print(
            f"Available valid flood pixels: "
            f"{len(flood_indices):,}"
        )

        if len(flood_indices) < N_FLOOD:
            raise RuntimeError(
                "Not enough valid recurrent "
                "flood pixels."
            )

        chosen = rng.choice(
            flood_indices,
            size=N_FLOOD,
            replace=False
        )

        flood_rows = (
            chosen // mask.width
        ).astype(np.int64)

        flood_cols = (
            chosen % mask.width
        ).astype(np.int64)


    print(
        f"✓ Selected "
        f"{len(flood_rows):,} unique flood pixels"
    )


    # ========================================================
    # 2. SAMPLE 10,000 CLEAN NON-FLOOD PIXELS
    # ========================================================

    print(
        "\n[2/4] Sampling clean non-flood pixels..."
    )

    any_flood_gdf = gpd.read_file(
        ANY_FLOOD
    )

    if any_flood_gdf.crs != ref.crs:
        any_flood_gdf = (
            any_flood_gdf.to_crs(ref.crs)
        )

    # Already dissolved into one MultiPolygon
    any_flood_geom = (
        any_flood_gdf.geometry.iloc[0]
    )

    # Speeds up repeated spatial queries
    prepare(any_flood_geom)


    selected_negative = set()

    attempts = 0


    while len(selected_negative) < N_NONFLOOD:

        attempts += 1

        # --------------------------------------------
        # Generate random candidate cells
        # --------------------------------------------

        rows = rng.integers(
            0,
            ref.height,
            size=BATCH_SIZE,
            dtype=np.int64
        )

        cols = rng.integers(
            0,
            ref.width,
            size=BATCH_SIZE,
            dtype=np.int64
        )


        # --------------------------------------------
        # Remove duplicates within this batch
        # --------------------------------------------

        flat = (
            rows * ref.width
            + cols
        )

        _, unique_idx = np.unique(
            flat,
            return_index=True
        )

        rows = rows[unique_idx]
        cols = cols[unique_idx]


        # --------------------------------------------
        # FIRST FILTER:
        # valid elevation
        # --------------------------------------------

        elevation = read_pixels(
            ref,
            rows,
            cols
        )

        valid = np.ones(
            len(rows),
            dtype=bool
        )

        if ref.nodata is not None:
            valid &= (
                elevation != ref.nodata
            )

        rows = rows[valid]
        cols = cols[valid]

        if len(rows) == 0:
            continue


        # --------------------------------------------
        # SECOND FILTER:
        # valid in ALL remaining FCFs
        # --------------------------------------------

        for feature, src in sources.items():

            if feature == "elevation":
                continue

            values = read_pixels(
                src,
                rows,
                cols
            )

            if src.nodata is not None:

                good = (
                    values != src.nodata
                )

                rows = rows[good]
                cols = cols[good]

            if len(rows) == 0:
                break


        if len(rows) == 0:
            continue


        # --------------------------------------------
        # Convert valid cell centres to coordinates
        # --------------------------------------------

        xs, ys = rasterio.transform.xy(
            ref.transform,
            rows,
            cols,
            offset="center"
        )

        xs = np.asarray(xs)
        ys = np.asarray(ys)


        # --------------------------------------------
        # THIRD FILTER:
        # exclude ALL known historical flood areas
        # --------------------------------------------

        inside_any_flood = contains_xy(
            any_flood_geom,
            xs,
            ys
        )

        keep = ~inside_any_flood

        rows = rows[keep]
        cols = cols[keep]


        # --------------------------------------------
        # Add only UNIQUE negative pixels
        # --------------------------------------------

        for r, c in zip(rows, cols):

            key = (
                int(r) * ref.width
                + int(c)
            )

            selected_negative.add(key)

            if (
                len(selected_negative)
                >= N_NONFLOOD
            ):
                break


        print(
            f"\rCollected "
            f"{len(selected_negative):,}"
            f"/{N_NONFLOOD:,} "
            f"clean negatives",
            end=""
        )


    print()

    negative_flat = np.fromiter(
        selected_negative,
        dtype=np.int64,
        count=N_NONFLOOD
    )

    negative_rows = (
        negative_flat // ref.width
    ).astype(np.int64)

    negative_cols = (
        negative_flat % ref.width
    ).astype(np.int64)


    print(
        f"✓ Selected "
        f"{len(negative_rows):,} "
        "unique clean non-flood pixels"
    )


    # ========================================================
    # COMBINE BOTH CLASSES
    # ========================================================

    rows = np.concatenate([
        flood_rows,
        negative_rows
    ])

    cols = np.concatenate([
        flood_cols,
        negative_cols
    ])

    labels = np.concatenate([
        np.ones(
            N_FLOOD,
            dtype=np.uint8
        ),
        np.zeros(
            N_NONFLOOD,
            dtype=np.uint8
        )
    ])


    # ========================================================
    # 3. EXTRACT ALL 10 FCF VALUES
    # ========================================================

    print(
        "\n[3/4] Extracting 10 FCFs..."
    )

    xs, ys = rasterio.transform.xy(
        ref.transform,
        rows,
        cols,
        offset="center"
    )

    data = {
        "class": labels,
        "row": rows,
        "col": cols,
        "x": np.asarray(xs),
        "y": np.asarray(ys),
    }


    for feature, src in sources.items():

        print(f"  {feature}...")

        stored = read_pixels(
            src,
            rows,
            cols
        )


        # --------------------------------------------
        # HARD NODATA CHECK
        # --------------------------------------------

        if src.nodata is not None:

            bad = (
                stored == src.nodata
            )

            if bad.any():

                raise RuntimeError(
                    f"{feature}: "
                    f"{bad.sum():,} sampled "
                    "NoData pixels found. "
                    "Sampling aborted."
                )


        # --------------------------------------------
        # RECONSTRUCT ORIGINAL / PHYSICAL VALUE
        # --------------------------------------------

        tags = src.tags()

        if (
            "SCALE_SRC_MIN" not in tags
            or "SCALE_SRC_MAX" not in tags
        ):
            raise RuntimeError(
                f"{feature}: scaling metadata "
                "not found."
            )

        src_min = float(
            tags["SCALE_SRC_MIN"]
        )

        src_max = float(
            tags["SCALE_SRC_MAX"]
        )

        physical = (
            src_min
            + (
                stored.astype(np.float64)
                / 65534.0
            )
            * (src_max - src_min)
        )

        data[feature] = physical


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(data)

feature_columns = list(
    FCFS.keys()
)


# ============================================================
# FINAL SAFETY CHECKS
# ============================================================

print("\nRunning final checks...")


# No missing values
if df[feature_columns].isna().any().any():
    raise RuntimeError(
        "NaN values detected in final dataset."
    )


# No infinities
if not np.isfinite(
    df[feature_columns].to_numpy()
).all():
    raise RuntimeError(
        "Infinite values detected "
        "in final dataset."
    )


# Exact class balance
flood_count = int(
    (df["class"] == 1).sum()
)

nonflood_count = int(
    (df["class"] == 0).sum()
)

if flood_count != N_FLOOD:
    raise RuntimeError(
        f"Expected {N_FLOOD} flood samples, "
        f"found {flood_count}."
    )

if nonflood_count != N_NONFLOOD:
    raise RuntimeError(
        f"Expected {N_NONFLOOD} non-flood samples, "
        f"found {nonflood_count}."
    )


# Ensure no duplicate grid cells
duplicates = df.duplicated(
    subset=["row", "col"]
).sum()

if duplicates != 0:
    raise RuntimeError(
        f"Found {duplicates} duplicate "
        "sample pixels."
    )


print("✓ No NoData values")
print("✓ No NaN or infinite values")
print("✓ No duplicate pixels")
print("✓ Exact 10,000 / 10,000 balance")


# ============================================================
# SAMPLE IDS
# ============================================================

df.insert(
    0,
    "sample_id",
    [
        f"S{i:05d}"
        for i in range(
            1,
            len(df) + 1
        )
    ]
)


# ============================================================
# 4. SAVE CSV + GPKG
# ============================================================

print("\n[4/4] Saving outputs...")


# Remove previous versions if they exist
if CSV_OUTPUT.exists():
    CSV_OUTPUT.unlink()

if GPKG_OUTPUT.exists():
    GPKG_OUTPUT.unlink()


df.to_csv(
    CSV_OUTPUT,
    index=False
)


geometry = [
    Point(x, y)
    for x, y in zip(
        df["x"],
        df["y"]
    )
]

gdf = gpd.GeoDataFrame(
    df,
    geometry=geometry,
    crs="EPSG:3395"
)

gdf.to_file(
    GPKG_OUTPUT,
    layer="training_samples",
    driver="GPKG"
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 80)
print("TRAINING SAMPLE COMPLETE")
print("=" * 80)

print(f"Total samples:     {len(df):,}")
print(f"Flood samples:     {flood_count:,}")
print(f"Non-flood samples: {nonflood_count:,}")

print("\nClass counts:")
print(
    df["class"]
    .value_counts()
    .sort_index()
)

print("\nPredictor ranges:")

for feature in feature_columns:

    print(
        f"  {feature:<22} "
        f"{df[feature].min():>12.4f}  to  "
        f"{df[feature].max():>12.4f}"
    )

print(f"\nCSV saved:\n{CSV_OUTPUT}")
print(f"\nGPKG saved:\n{GPKG_OUTPUT}")