from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely import make_valid


# ============================================================
# PATHS
# ============================================================

GDB = Path("Data/inventory/FL20100802PAK.gdb")

OUT_DIR = Path("Data/inventory/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_OUTPUT = OUT_DIR / "flood_2010_source_polygons.gpkg"
FINAL_OUTPUT = OUT_DIR / "flood_extent_2010.gpkg"

TARGET_CRS = "EPSG:3395"


# ============================================================
# UNOSAT 2010 FLOOD LAYERS
# ============================================================

LAYERS = [
    "QB_20100824_Flood",
    "DMC_20101006_Flood",
    "MOD_20100816_Flood",
    "MOD_20100812_Flood",
    "MOD_20100811_Flood",
    "MOD_20100810_Flood",
    "MOD_20100801_Flood",
    "MOD_20100808_Flood",
    "SPOT5_20100811_Flood",
    "ASAR_20100901_Flood",
    "ASAR_20100824_Flood",
    "RS2_20100810_Flood",
    "RS2_20100805_Flood",
    "ALOS_20100805_Flood",
    "ALOS_20100819_Flood",
    "MOD_20101206_Flood",
    "MOD_20101125_Flood",
    "MOD_20100824_Flood",
    "MOD_20100822_Flood",
    "MOD_20100818_Flood",
]


print("=" * 80)
print("BUILDING 2010 UNOSAT FLOOD INVENTORY")
print("=" * 80)

frames = []


# ============================================================
# READ EACH FLOOD LAYER
# ============================================================

for layer in LAYERS:

    print(f"\nReading {layer}...")

    gdf = gpd.read_file(
        GDB,
        layer=layer
    )

    original_count = len(gdf)

    # --------------------------------------------------------
    # STRICT POSITIVE CLASS
    # Water_Class = 1 = confirmed Flood Water
    # --------------------------------------------------------

    if "Water_Class" not in gdf.columns:
        print("  Skipped: no Water_Class field")
        continue

    gdf = gdf[
        gdf["Water_Class"].eq(1)
        & gdf.geometry.notna()
        & ~gdf.geometry.is_empty
    ].copy()

    print(f"  Original polygons: {original_count:,}")
    print(f"  Confirmed flood polygons: {len(gdf):,}")

    if len(gdf) == 0:
        print("  No class-1 flood polygons — skipped")
        continue


    # --------------------------------------------------------
    # REPAIR INVALID GEOMETRIES
    # --------------------------------------------------------

    invalid = ~gdf.geometry.is_valid

    if invalid.any():

        print(
            f"  Repairing {invalid.sum():,} "
            "invalid geometries..."
        )

        gdf.loc[invalid, "geometry"] = (
            gdf.loc[invalid, "geometry"]
            .apply(make_valid)
        )


    # --------------------------------------------------------
    # REPROJECT
    # --------------------------------------------------------

    gdf = gdf.to_crs(TARGET_CRS)

    gdf["source_layer"] = layer

    frames.append(
        gdf[["source_layer", "geometry"]]
    )


# ============================================================
# COMBINE
# ============================================================

combined = gpd.GeoDataFrame(
    pd.concat(
        frames,
        ignore_index=True
    ),
    crs=TARGET_CRS
)

print("\n" + "=" * 80)
print("COMBINED 2010 INVENTORY")
print("=" * 80)

print(
    f"Confirmed flood polygons before union: "
    f"{len(combined):,}"
)


# ============================================================
# SAVE TRACEABLE SOURCE POLYGONS
# ============================================================

combined.to_file(
    SOURCE_OUTPUT,
    layer="flood_2010_sources",
    driver="GPKG"
)

print(
    f"\nSource polygons saved:\n"
    f"{SOURCE_OUTPUT}"
)


# ============================================================
# DISSOLVE OVERLAPS
# ============================================================

print("\nDissolving overlapping polygons...")
print("This may take a few minutes.")

merged_geometry = combined.geometry.union_all()


# ============================================================
# FINAL INVENTORY
# ============================================================

final = gpd.GeoDataFrame(
    {
        "year": [2010],
        "source": ["UNOSAT"],
        "class": ["Confirmed Flood Water"],
    },
    geometry=[merged_geometry],
    crs=TARGET_CRS
)


final.to_file(
    FINAL_OUTPUT,
    layer="flood_2010",
    driver="GPKG"
)


# ============================================================
# FINAL CHECK
# ============================================================

print("\n" + "=" * 80)
print("2010 INVENTORY COMPLETE")
print("=" * 80)

print(f"CRS: {final.crs}")
print(
    f"Geometry type: "
    f"{final.geometry.geom_type.iloc[0]}"
)

print(f"Bounds: {final.total_bounds}")

print(
    f"\nFinal output:\n"
    f"{FINAL_OUTPUT}"
)