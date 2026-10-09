from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely import make_valid


# PATHS

GDB = Path("Data/inventory/FL20140910PAK.gdb")

OUT_DIR = Path("Data/inventory/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_OUTPUT = OUT_DIR / "flood_2014_source_polygons.gpkg"
FINAL_OUTPUT = OUT_DIR / "flood_extent_2014.gpkg"


# 2014 UNOSAT FLOOD LAYERS

LAYERS = [
    "LS_20140910_Flood",
    "TX_20140915_Flood",
    "TX_20140916_Flood",
    "SN1_20140916_Flood",
    "TSX_20140916_Flood",
]

# Same CRS as the flood-conditioning rasters
TARGET_CRS = "EPSG:3395"


print("=" * 80)
print("BUILDING 2014 UNOSAT FLOOD INVENTORY")
print("=" * 80)


# READ + CLEAN EACH FLOOD LAYER

frames = []

for layer in LAYERS:

    print(f"\nReading {layer}...")

    gdf = gpd.read_file(
        GDB,
        layer=layer
    )

    print(f"  Original polygons: {len(gdf):,}")

    # Keep geometry only
    gdf = gdf[["geometry"]].copy()

    # Remove missing / empty geometries
    gdf = gdf[
        gdf.geometry.notna()
        & ~gdf.geometry.is_empty
    ].copy()

    # Repair invalid geometries if necessary
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

    # Reproject to FCF CRS
    gdf = gdf.to_crs(TARGET_CRS)

    # Track where each polygon came from
    gdf["source_layer"] = layer

    frames.append(
        gdf[["source_layer", "geometry"]]
    )

    print(f"  Retained polygons: {len(gdf):,}")


# COMBINE ALL FIVE SENSOR FLOOD LAYERS

combined = gpd.GeoDataFrame(
    pd.concat(
        frames,
        ignore_index=True
    ),
    crs=TARGET_CRS
)

print("\n" + "=" * 80)
print("COMBINED INVENTORY")
print("=" * 80)

print(
    f"Total polygons before union: "
    f"{len(combined):,}"
)


# SAVE SOURCE POLYGONS

combined.to_file(
    SOURCE_OUTPUT,
    layer="flood_2014_sources",
    driver="GPKG"
)

print(
    f"\nSource polygons saved:\n"
    f"{SOURCE_OUTPUT}"
)


# UNION / DISSOLVE OVERLAPS

print("\nDissolving overlapping polygons...")
print("This may take a few minutes.")

merged_geometry = combined.geometry.union_all()


# CREATE FINAL 2014 INVENTORY

final = gpd.GeoDataFrame(
    {
        "year": [2014],
        "source": ["UNOSAT"],
        "event": ["Pakistan Flood 2014"],
    },
    geometry=[merged_geometry],
    crs=TARGET_CRS
)


# SAVE FINAL LAYER

final.to_file(
    FINAL_OUTPUT,
    layer="flood_2014",
    driver="GPKG"
)


# FINAL CHECK

print("\n" + "=" * 80)
print("2014 INVENTORY COMPLETE")
print("=" * 80)

print(f"CRS: {final.crs}")
print(
    f"Geometry type: "
    f"{final.geometry.geom_type.iloc[0]}"
)

print(
    f"Bounds:\n"
    f"{final.total_bounds}"
)

print(
    f"\nFinal output:\n"
    f"{FINAL_OUTPUT}"
)