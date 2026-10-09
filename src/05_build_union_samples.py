from pathlib import Path
from contextlib import ExitStack
import math

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

from pyproj import Transformer
from rasterio.windows import Window, from_bounds
from shapely import contains_xy, prepare
from shapely.geometry import Point


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42

N_FLOOD = 10_000
N_BACKGROUND = 10_000

# Build a much larger candidate pool first.
# Final 10k samples are then selected spatially.
CANDIDATE_POOL_SIZE = 60_000

BATCH_SIZE = 50_000

# Spatial sampling grid
SPATIAL_BLOCK_M = 100_000       # 100 km

# Give geographic coverage without making every block equal
MIN_PER_BLOCK = 20

# Prevent one large floodplain block dominating the sample
MAX_PER_BLOCK = 300


# ============================================================
# PATHS
# ============================================================

FCF_DIR = Path("Data/FCF")

UNION_VECTOR = Path(
    "Data/inventory/processed/"
    "any_flood_2010_2014_2022.gpkg"
)

OUT_DIR = Path("Data/samples")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_OUTPUT = (
    OUT_DIR /
    "training_samples_union_raw.csv"
)

GPKG_OUTPUT = (
    OUT_DIR /
    "training_samples_union.gpkg"
)


# ============================================================
# FCFs
# ============================================================

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
print("SPATIALLY DISTRIBUTED UNION-BASED FSM SAMPLING")
print("=" * 80)


# ============================================================
# READ ARBITRARY RASTER CELLS EFFICIENTLY
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
# KEEP PIXELS VALID IN ALL 10 FCFs
# ============================================================

def keep_valid_fcfs(
    rows,
    cols,
    sources
):

    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)

    for feature, src in sources.items():

        if len(rows) == 0:
            break

        values = read_pixels(
            src,
            rows,
            cols
        )

        nodata = src.nodata

        if nodata is not None:

            good = values != nodata

            rows = rows[good]
            cols = cols[good]

    return rows, cols


# ============================================================
# ASSIGN 100-km BLOCK
# ============================================================

def add_spatial_blocks(
    df,
    ref_crs
):

    transformer = Transformer.from_crs(
        ref_crs,
        "EPSG:6933",
        always_xy=True
    )

    x_ea, y_ea = transformer.transform(
        df["x"].to_numpy(),
        df["y"].to_numpy()
    )

    block_x = np.floor(
        np.asarray(x_ea)
        / SPATIAL_BLOCK_M
    ).astype(int)

    block_y = np.floor(
        np.asarray(y_ea)
        / SPATIAL_BLOCK_M
    ).astype(int)

    df["sampling_block"] = [
        f"B_{bx}_{by}"
        for bx, by in zip(
            block_x,
            block_y
        )
    ]

    return df


# ============================================================
# CREATE LARGE RANDOM CANDIDATE POOL
# ============================================================

def build_candidate_pool(
    class_value,
    target_size,
    ref,
    sources,
    flood_geom,
    flood_bounds
):

    selected = {}

    if class_value == 1:

        minx, miny, maxx, maxy = flood_bounds

        w = from_bounds(
            minx,
            miny,
            maxx,
            maxy,
            transform=ref.transform
        )

        row_min = max(
            0,
            int(np.floor(w.row_off))
        )

        row_max = min(
            ref.height,
            int(
                np.ceil(
                    w.row_off + w.height
                )
            )
        )

        col_min = max(
            0,
            int(np.floor(w.col_off))
        )

        col_max = min(
            ref.width,
            int(
                np.ceil(
                    w.col_off + w.width
                )
            )
        )

    else:

        row_min = 0
        row_max = ref.height
        col_min = 0
        col_max = ref.width


    label = (
        "Flood"
        if class_value == 1
        else "Background"
    )

    print(
        f"\nBuilding {label} candidate pool..."
    )


    while len(selected) < target_size:

        rows = rng.integers(
            row_min,
            row_max,
            size=BATCH_SIZE,
            dtype=np.int64
        )

        cols = rng.integers(
            col_min,
            col_max,
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
        # Pixel centre coordinates
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
        # Flood / background membership
        # --------------------------------------------

        inside = contains_xy(
            flood_geom,
            xs,
            ys
        )

        if class_value == 1:
            keep = inside
        else:
            keep = ~inside


        rows = rows[keep]
        cols = cols[keep]


        if len(rows) == 0:
            continue


        # --------------------------------------------
        # Require ALL FCFs valid
        # --------------------------------------------

        rows, cols = keep_valid_fcfs(
            rows,
            cols,
            sources
        )


        # --------------------------------------------
        # Save unique candidates
        # --------------------------------------------

        for r, c in zip(rows, cols):

            key = (
                int(r) * ref.width
                + int(c)
            )

            if key not in selected:

                selected[key] = (
                    int(r),
                    int(c)
                )

            if len(selected) >= target_size:
                break


        print(
            f"\r  {label}: "
            f"{len(selected):,}"
            f"/{target_size:,}",
            end=""
        )


    print()


    values = list(selected.values())

    rows = np.asarray(
        [v[0] for v in values],
        dtype=np.int64
    )

    cols = np.asarray(
        [v[1] for v in values],
        dtype=np.int64
    )


    xs, ys = rasterio.transform.xy(
        ref.transform,
        rows,
        cols,
        offset="center"
    )


    pool = pd.DataFrame({
        "row": rows,
        "col": cols,
        "x": np.asarray(xs),
        "y": np.asarray(ys),
    })


    pool = add_spatial_blocks(
        pool,
        ref.crs
    )

    print(
        f"  Occupied 100-km blocks: "
        f"{pool['sampling_block'].nunique():,}"
    )

    return pool


# ============================================================
# SPATIALLY DISTRIBUTED SELECTION
#
# 1. Small minimum from occupied blocks
# 2. Remaining points selected randomly
# 3. Larger blocks naturally contribute more
# 4. No block can exceed cap
# ============================================================

def spatial_select(
    pool,
    n_required,
    label
):

    pool = pool.copy()

    groups = {
        block: group.index.to_numpy()
        for block, group
        in pool.groupby("sampling_block")
    }

    n_blocks = len(groups)

    if n_blocks == 0:
        raise RuntimeError(
            f"No spatial blocks found for {label}."
        )


    print(
        f"\nSpatially selecting {label}:"
    )

    print(
        f"  Candidate blocks: {n_blocks:,}"
    )


    # --------------------------------------------
    # Effective minimum
    # --------------------------------------------

    effective_min = min(
        MIN_PER_BLOCK,
        max(
            1,
            n_required // n_blocks
        )
    )


    # --------------------------------------------
    # Make sure cap allows enough total capacity
    # --------------------------------------------

    counts = (
        pool["sampling_block"]
        .value_counts()
    )


    effective_cap = max(
        MAX_PER_BLOCK,
        math.ceil(
            n_required / n_blocks
        )
    )


    while (
        np.minimum(
            counts.to_numpy(),
            effective_cap
        ).sum()
        < n_required
    ):

        effective_cap += 25


    print(
        f"  Minimum/block: {effective_min}"
    )

    print(
        f"  Maximum/block: {effective_cap}"
    )


    selected_indices = []
    selected_set = set()
    selected_per_block = {
        block: 0
        for block in groups
    }


    # ========================================================
    # STAGE 1:
    # minimum coverage in each occupied block
    # ========================================================

    block_order = list(groups.keys())
    rng.shuffle(block_order)


    for block in block_order:

        indices = groups[block]

        n_take = min(
            effective_min,
            len(indices),
            effective_cap
        )

        if n_take <= 0:
            continue


        chosen = rng.choice(
            indices,
            size=n_take,
            replace=False
        )


        for idx in chosen:

            idx = int(idx)

            selected_indices.append(idx)
            selected_set.add(idx)

        selected_per_block[block] += n_take


    # If somehow minimum stage exceeds target
    if len(selected_indices) > n_required:

        chosen = rng.choice(
            np.asarray(selected_indices),
            size=n_required,
            replace=False
        )

        result = pool.loc[
            chosen
        ].copy()

        return result.reset_index(drop=True)


    # ========================================================
    # STAGE 2:
    # proportional random fill with a cap
    #
    # Because candidate pool itself is randomly drawn from
    # eligible pixels, shuffling the remaining candidates
    # approximately preserves area proportionality.
    # ========================================================

    remaining_indices = np.asarray(
        [
            idx
            for idx in pool.index
            if idx not in selected_set
        ],
        dtype=np.int64
    )

    rng.shuffle(
        remaining_indices
    )


    for idx in remaining_indices:

        if len(selected_indices) >= n_required:
            break

        block = pool.at[
            idx,
            "sampling_block"
        ]

        if (
            selected_per_block[block]
            >= effective_cap
        ):
            continue


        selected_indices.append(
            int(idx)
        )

        selected_per_block[block] += 1


    if len(selected_indices) < n_required:

        raise RuntimeError(
            f"Could only select "
            f"{len(selected_indices):,} "
            f"{label} samples. "
            f"Increase CANDIDATE_POOL_SIZE."
        )


    result = pool.loc[
        selected_indices[:n_required]
    ].copy()


    result = result.reset_index(
        drop=True
    )


    counts_final = (
        result["sampling_block"]
        .value_counts()
    )


    print(
        f"  Final samples: "
        f"{len(result):,}"
    )

    print(
        f"  Final occupied blocks: "
        f"{result['sampling_block'].nunique():,}"
    )

    print(
        f"  Largest block: "
        f"{counts_final.iloc[0]:,} "
        f"({100 * counts_final.iloc[0] / len(result):.2f}%)"
    )


    return result


# ============================================================
# LOAD UNION
# ============================================================

print(
    "\nLoading historical flood union..."
)

union = gpd.read_file(
    UNION_VECTOR
)


# ============================================================
# OPEN ALL FCFs
# ============================================================

with ExitStack() as stack:

    sources = {
        feature:
        stack.enter_context(
            rasterio.open(
                FCF_DIR / filename
            )
        )
        for feature, filename
        in FCFS.items()
    }

    ref = sources["elevation"]


    # ========================================================
    # ALIGNMENT CHECK
    # ========================================================

    print(
        "\nChecking FCF alignment..."
    )

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

    print(
        "✓ All 10 FCFs aligned"
    )


    # ========================================================
    # PREPARE UNION
    # ========================================================

    if union.crs != ref.crs:

        union = union.to_crs(
            ref.crs
        )


    flood_geom = (
        union.geometry.union_all()
    )

    prepare(
        flood_geom
    )

    flood_bounds = union.total_bounds


    print(
        "✓ Historical flood union loaded"
    )


    # ========================================================
    # BUILD CANDIDATE POOLS
    # ========================================================

    flood_pool = build_candidate_pool(
        class_value=1,
        target_size=CANDIDATE_POOL_SIZE,
        ref=ref,
        sources=sources,
        flood_geom=flood_geom,
        flood_bounds=flood_bounds
    )


    background_pool = build_candidate_pool(
        class_value=0,
        target_size=CANDIDATE_POOL_SIZE,
        ref=ref,
        sources=sources,
        flood_geom=flood_geom,
        flood_bounds=flood_bounds
    )


    # ========================================================
    # SPATIALLY DISTRIBUTED FINAL DRAW
    # ========================================================

    flood_sample = spatial_select(
        flood_pool,
        N_FLOOD,
        "historical flood"
    )


    background_sample = spatial_select(
        background_pool,
        N_BACKGROUND,
        "background"
    )


    flood_sample["class"] = 1
    background_sample["class"] = 0


    sample = pd.concat(
        [
            flood_sample,
            background_sample
        ],
        ignore_index=True
    )


    # ========================================================
    # EXTRACT FCF VALUES
    # ========================================================

    print(
        "\nExtracting FCF values "
        "for final 20,000 samples..."
    )


    rows = sample["row"].to_numpy(
        dtype=np.int64
    )

    cols = sample["col"].to_numpy(
        dtype=np.int64
    )


    for feature, src in sources.items():

        print(
            f"  {feature}..."
        )

        stored = read_pixels(
            src,
            rows,
            cols
        )


        if (
            src.nodata is not None
            and np.any(
                stored == src.nodata
            )
        ):

            raise RuntimeError(
                f"{feature}: NoData found "
                "in final samples."
            )


        tags = src.tags()


        if (
            "SCALE_SRC_MIN" not in tags
            or "SCALE_SRC_MAX" not in tags
        ):

            raise RuntimeError(
                f"{feature}: scaling tags missing."
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
            * (
                src_max - src_min
            )
        )


        sample[feature] = physical


# ============================================================
# FINAL DATASET
# ============================================================

df = sample.copy()


FEATURE_COLUMNS = list(
    FCFS.keys()
)


# ============================================================
# SAFETY CHECKS
# ============================================================

print(
    "\nRunning final checks..."
)


if len(df) != 20_000:

    raise RuntimeError(
        f"Expected 20,000 rows, "
        f"found {len(df):,}"
    )


if (
    df[FEATURE_COLUMNS]
    .isna()
    .any()
    .any()
):

    raise RuntimeError(
        "NaN values found."
    )


if not np.isfinite(
    df[FEATURE_COLUMNS].to_numpy()
).all():

    raise RuntimeError(
        "Infinite values found."
    )


duplicate_cells = df.duplicated(
    subset=["row", "col"]
).sum()


if duplicate_cells > 0:

    raise RuntimeError(
        f"{duplicate_cells:,} "
        "duplicate raster cells found."
    )


flood_count = int(
    (df["class"] == 1).sum()
)

background_count = int(
    (df["class"] == 0).sum()
)


if (
    flood_count != N_FLOOD
    or background_count != N_BACKGROUND
):

    raise RuntimeError(
        "Class balance is incorrect."
    )


print("✓ 20,000 total samples")
print("✓ 10,000 flood")
print("✓ 10,000 background")
print("✓ No duplicate cells")
print("✓ No NaN / infinity")
print("✓ All 10 FCFs valid")


# ============================================================
# CLASS LABEL
# ============================================================

df.insert(
    0,
    "sample_id",
    [
        f"U{i:05d}"
        for i in range(
            1,
            len(df) + 1
        )
    ]
)


df.insert(
    2,
    "class_name",
    np.where(
        df["class"] == 1,
        "historical_flood",
        "background"
    )
)


# ============================================================
# LATITUDE QC
# ============================================================

to_lonlat = Transformer.from_crs(
    "EPSG:3395",
    "EPSG:4326",
    always_xy=True
)


longitude, latitude = (
    to_lonlat.transform(
        df["x"].to_numpy(),
        df["y"].to_numpy()
    )
)

longitude = np.asarray(longitude)
latitude = np.asarray(latitude)


# ============================================================
# SAVE
# ============================================================

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
print("FINAL SPATIALLY DISTRIBUTED UNION SAMPLE")
print("=" * 80)


print(
    f"\nHistorical flood: "
    f"{flood_count:,}"
)

print(
    f"Background:       "
    f"{background_count:,}"
)


# ============================================================
# MEDIANS
# ============================================================

print(
    "\nPredictor medians:"
)


for feature in FEATURE_COLUMNS:

    bg = df.loc[
        df["class"] == 0,
        feature
    ].median()

    fl = df.loc[
        df["class"] == 1,
        feature
    ].median()


    print(
        f"  {feature:<22}"
        f" background={bg:>12.4f}"
        f"   flood={fl:>12.4f}"
    )


# ============================================================
# SPATIAL QC
# ============================================================

print(
    "\nSpatial sampling coverage:"
)


for cls, name in [
    (1, "Historical flood"),
    (0, "Background")
]:

    subset = df[
        df["class"] == cls
    ]

    counts = (
        subset[
            "sampling_block"
        ]
        .value_counts()
    )


    print(
        f"\n{name}"
    )

    print(
        f"  Occupied 100-km blocks: "
        f"{len(counts):,}"
    )

    print(
        f"  Largest block: "
        f"{counts.iloc[0]:,} "
        f"({100 * counts.iloc[0] / len(subset):.2f}%)"
    )


# ============================================================
# NORTH/SOUTH QC
# ============================================================

print(
    "\nLatitude coverage:"
)


for cls, name in [
    (1, "Historical flood"),
    (0, "Background")
]:

    mask = (
        df["class"].to_numpy()
        == cls
    )

    lat = latitude[mask]


    print(
        f"\n{name}"
    )

    print(
        f"  Latitude range: "
        f"{lat.min():.2f}° "
        f"to {lat.max():.2f}°"
    )

    print(
        f"  Samples south of 27°N: "
        f"{np.sum(lat < 27):,}"
    )


print(
    f"\nCSV:\n{CSV_OUTPUT}"
)

print(
    f"\nGPKG:\n{GPKG_OUTPUT}"
)