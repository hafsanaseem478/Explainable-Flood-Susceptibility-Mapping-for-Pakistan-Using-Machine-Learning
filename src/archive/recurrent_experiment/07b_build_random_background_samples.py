from pathlib import Path
from contextlib import ExitStack

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from shapely import contains_xy, prepare
from shapely.geometry import Point


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42
N_NONFLOOD = 10_000
BATCH_SIZE = 30_000


# ============================================================
# PATHS
# ============================================================

FCF_DIR = Path("Data/FCF")

# Existing clean dataset:
# We reuse its EXACT SAME 10,000 positive pixels
STRICT_SAMPLE = Path(
    "Data/samples/training_samples_raw.csv"
)

RECURRENT_VECTOR = Path(
    "Data/inventory/processed/"
    "recurrent_flood_2010_2014_2022.gpkg"
)

# Used ONLY to report how many new negatives lie within
# one or more historical flood footprints.
ANY_FLOOD = Path(
    "Data/inventory/processed/"
    "any_flood_2010_2014_2022.gpkg"
)

OUT_DIR = Path("Data/samples")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_OUTPUT = (
    OUT_DIR
    / "training_samples_random_background_raw.csv"
)

GPKG_OUTPUT = (
    OUT_DIR
    / "training_samples_random_background.gpkg"
)


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
print("BUILDING RANDOM-BACKGROUND TRAINING SAMPLE")
print("=" * 80)


# ============================================================
# HELPER: READ SPECIFIC PIXELS
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
# 1. LOAD SAME POSITIVE POINTS
# ============================================================

print("\n[1/4] Loading existing recurrent-flood samples...")

old = pd.read_csv(STRICT_SAMPLE)

positives = old[
    old["class"] == 1
].copy()

if len(positives) != 10_000:
    raise RuntimeError(
        f"Expected 10,000 existing positives, "
        f"found {len(positives):,}"
    )

if positives.duplicated(
    subset=["row", "col"]
).any():
    raise RuntimeError(
        "Duplicate positive pixels detected."
    )

print(
    "✓ Reusing exactly "
    f"{len(positives):,} flood pixels"
)


# ============================================================
# OPEN ALL FCF RASTERS
# ============================================================

with ExitStack() as stack:

    sources = {
        feature: stack.enter_context(
            rasterio.open(FCF_DIR / filename)
        )
        for feature, filename in FCFS.items()
    }

    ref = sources["elevation"]


    # ========================================================
    # GRID ALIGNMENT CHECK
    # ========================================================

    for feature, src in sources.items():

        if (
            src.crs != ref.crs
            or src.width != ref.width
            or src.height != ref.height
            or src.transform != ref.transform
        ):
            raise RuntimeError(
                f"{feature} is not aligned "
                "with elevation.tif."
            )

    print("✓ All 10 FCF rasters aligned")


    # ========================================================
    # RECURRENT POSITIVE GEOMETRY
    # ========================================================

    recurrent = gpd.read_file(
        RECURRENT_VECTOR
    )

    if recurrent.crs != ref.crs:
        recurrent = recurrent.to_crs(
            ref.crs
        )

    recurrent_geom = (
        recurrent.geometry.iloc[0]
    )

    prepare(recurrent_geom)


    # ========================================================
    # 2. SAMPLE RANDOM BACKGROUND NEGATIVES
    # ========================================================

    print(
        "\n[2/4] Sampling random-background "
        "non-flood pixels..."
    )

    selected_negative = set()

    while len(selected_negative) < N_NONFLOOD:

        # --------------------------------------------
        # Random national grid cells
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
        # Remove duplicate candidates
        # --------------------------------------------

        flat = (
            rows * ref.width
            + cols
        )

        _, idx = np.unique(
            flat,
            return_index=True
        )

        rows = rows[idx]
        cols = cols[idx]


        # --------------------------------------------
        # Require valid data in ALL 10 FCFs
        # --------------------------------------------

        valid = np.ones(
            len(rows),
            dtype=bool
        )

        for feature, src in sources.items():

            values = read_pixels(
                src,
                rows,
                cols
            )

            if src.nodata is not None:

                valid &= (
                    values != src.nodata
                )

        rows = rows[valid]
        cols = cols[valid]

        if len(rows) == 0:
            continue


        # --------------------------------------------
        # Pixel-centre coordinates
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
        # EXCLUDE ONLY THE RECURRENT POSITIVE AREA
        #
        # Important:
        # We do NOT exclude the union of 2010/2014/2022.
        # --------------------------------------------

        inside_recurrent = contains_xy(
            recurrent_geom,
            xs,
            ys
        )

        keep = ~inside_recurrent

        rows = rows[keep]
        cols = cols[keep]


        # --------------------------------------------
        # Store unique cells
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
            f"/{N_NONFLOOD:,}",
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
        "random-background negatives"
    )


    # ========================================================
    # EXTRACT NEGATIVE FCF VALUES
    # ========================================================

    print(
        "\n[3/4] Extracting FCF values "
        "for new negatives..."
    )

    xs, ys = rasterio.transform.xy(
        ref.transform,
        negative_rows,
        negative_cols,
        offset="center"
    )

    negative_data = {
        "class":
            np.zeros(
                N_NONFLOOD,
                dtype=np.uint8
            ),

        "row":
            negative_rows,

        "col":
            negative_cols,

        "x":
            np.asarray(xs),

        "y":
            np.asarray(ys),
    }


    for feature, src in sources.items():

        print(f"  {feature}...")

        stored = read_pixels(
            src,
            negative_rows,
            negative_cols
        )


        # HARD NODATA CHECK
        if src.nodata is not None:

            bad = stored == src.nodata

            if bad.any():
                raise RuntimeError(
                    f"{feature}: "
                    f"{bad.sum():,} NoData "
                    "pixels found."
                )


        # Restore physical units
        tags = src.tags()

        if (
            "SCALE_SRC_MIN" not in tags
            or "SCALE_SRC_MAX" not in tags
        ):
            raise RuntimeError(
                f"{feature}: scaling "
                "metadata missing."
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

        negative_data[
            feature
        ] = physical


# ============================================================
# CREATE NEW NEGATIVE DATAFRAME
# ============================================================

negatives = pd.DataFrame(
    negative_data
)


# ============================================================
# USE SAME COLUMN STRUCTURE AS ORIGINAL POSITIVES
# ============================================================

FEATURES = list(FCFS.keys())

base_columns = [
    "class",
    "row",
    "col",
    "x",
    "y",
] + FEATURES


positives = positives[
    base_columns
].copy()

negatives = negatives[
    base_columns
].copy()


# ============================================================
# COMBINE
# ============================================================

df = pd.concat(
    [
        positives,
        negatives,
    ],
    ignore_index=True
)


# ============================================================
# FINAL SAFETY CHECKS
# ============================================================

print("\nRunning final checks...")

if len(df) != 20_000:
    raise RuntimeError(
        f"Expected 20,000 rows, "
        f"found {len(df):,}"
    )

if df[FEATURES].isna().any().any():
    raise RuntimeError(
        "NaN detected."
    )

if not np.isfinite(
    df[FEATURES].to_numpy()
).all():
    raise RuntimeError(
        "Infinite values detected."
    )

duplicates = df.duplicated(
    subset=["row", "col"]
).sum()

if duplicates:
    raise RuntimeError(
        f"{duplicates} duplicate "
        "grid cells detected."
    )

if (
    (df["class"] == 1).sum()
    != 10_000
):
    raise RuntimeError(
        "Incorrect positive count."
    )

if (
    (df["class"] == 0).sum()
    != 10_000
):
    raise RuntimeError(
        "Incorrect negative count."
    )


print("✓ 20,000 samples")
print("✓ 10,000 flood")
print("✓ 10,000 random-background")
print("✓ No NoData")
print("✓ No duplicates")


# ============================================================
# HOW MANY NEW NEGATIVES ARE INSIDE ANY HISTORICAL FLOOD?
# ============================================================

print(
    "\nChecking historical-flood overlap "
    "among new negatives..."
)

any_flood = gpd.read_file(
    ANY_FLOOD
)

if any_flood.crs != "EPSG:3395":
    any_flood = any_flood.to_crs(
        "EPSG:3395"
    )

any_flood_geom = (
    any_flood.geometry.iloc[0]
)

prepare(any_flood_geom)


inside_any = contains_xy(
    any_flood_geom,
    negatives["x"].to_numpy(),
    negatives["y"].to_numpy()
)

n_inside_any = int(
    inside_any.sum()
)

print(
    f"Negatives located inside at least one "
    f"2010/2014/2022 flood footprint: "
    f"{n_inside_any:,}"
)

print(
    f"Percentage of negatives: "
    f"{100 * n_inside_any / N_NONFLOOD:.2f}%"
)


# ============================================================
# NEW SAMPLE IDs
# ============================================================

df.insert(
    0,
    "sample_id",
    [
        f"RB{i:05d}"
        for i in range(
            1,
            len(df) + 1
        )
    ]
)


# ============================================================
# SAVE
# ============================================================

print("\n[4/4] Saving...")


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
# REPORT
# ============================================================

print("\n" + "=" * 80)
print("RANDOM-BACKGROUND SAMPLE COMPLETE")
print("=" * 80)

print(f"Total samples:     {len(df):,}")
print(
    f"Flood samples:     "
    f"{(df['class'] == 1).sum():,}"
)
print(
    f"Background samples:"
    f" {(df['class'] == 0).sum():,}"
)

print(
    f"\nBackground points inside "
    f"any historical flood footprint: "
    f"{n_inside_any:,}"
)

print(
    f"Share: "
    f"{100 * n_inside_any / N_NONFLOOD:.2f}%"
)

print(
    f"\nCSV:\n{CSV_OUTPUT}"
)

print(
    f"\nGPKG:\n{GPKG_OUTPUT}"
)