from pathlib import Path

import geopandas as gpd
from shapely import make_valid


# ============================================================
# PATHS
# ============================================================

INPUT = Path(
    "Data/inventory/FL20220808PAK_SHP/"
    "VIIRS_20220701_20220831_FloodExtent_PAK.shp"
)

OUT_DIR = Path("Data/inventory/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "flood_extent_2022.gpkg"

TARGET_CRS = "EPSG:3395"


# ============================================================
# READ
# ============================================================

print("=" * 80)
print("PREPARING 2022 UNOSAT FLOOD INVENTORY")
print("=" * 80)

gdf = gpd.read_file(INPUT)

print(f"\nFeatures: {len(gdf):,}")
print(f"Original CRS: {gdf.crs}")
print(f"Geometry: {gdf.geometry.geom_type.value_counts().to_dict()}")

if "Water_Clas" in gdf.columns:
    print(
        "Water class:",
        gdf["Water_Clas"].dropna().unique()
    )


# ============================================================
# CLEAN
# ============================================================

gdf = gdf[
    gdf.geometry.notna()
    & ~gdf.geometry.is_empty
].copy()

invalid = ~gdf.geometry.is_valid

if invalid.any():
    print(f"Repairing {invalid.sum():,} invalid geometries...")
    gdf.loc[invalid, "geometry"] = (
        gdf.loc[invalid, "geometry"].apply(make_valid)
    )


# ============================================================
# REPROJECT TO FCF CRS
# ============================================================

gdf = gdf.to_crs(TARGET_CRS)

merged_geometry = gdf.geometry.union_all()


# ============================================================
# FINAL INVENTORY
# ============================================================

final = gpd.GeoDataFrame(
    {
        "year": [2022],
        "source": ["UNOSAT"],
        "class": ["Flood Water"],
        "period": ["2022-07-01 to 2022-08-31"],
    },
    geometry=[merged_geometry],
    crs=TARGET_CRS
)

final.to_file(
    OUTPUT,
    layer="flood_2022",
    driver="GPKG"
)


# REPORT

print("\n" + "=" * 80)
print("2022 INVENTORY COMPLETE")
print("=" * 80)

print(f"CRS: {final.crs}")
print(f"Geometry type: {final.geometry.geom_type.iloc[0]}")
print(f"Bounds: {final.total_bounds}")

print(f"\nFinal output:\n{OUTPUT}")