from pathlib import Path

import geopandas as gpd
from shapely import make_valid


# ============================================================
# PATHS
# ============================================================

BASE = Path("Data/inventory/processed")

FILES = [
    BASE / "flood_extent_2010.gpkg",
    BASE / "flood_extent_2014.gpkg",
    BASE / "flood_extent_2022.gpkg",
]

OUTPUT = BASE / "any_flood_2010_2014_2022.gpkg"

TARGET_CRS = "EPSG:3395"
AREA_CRS = "EPSG:6933"


print("=" * 80)
print("BUILDING ANY-HISTORICAL-FLOOD INVENTORY")
print("=" * 80)


# ============================================================
# READ EVENT INVENTORIES
# ============================================================

geometries = []

for file in FILES:

    gdf = gpd.read_file(file)

    if gdf.crs != TARGET_CRS:
        gdf = gdf.to_crs(TARGET_CRS)

    geom = gdf.geometry.union_all()

    if not geom.is_valid:
        geom = make_valid(geom)

    geometries.append(geom)

    print(f"Loaded: {file.name}")


# ============================================================
# UNION
# ============================================================

print("\nCombining all historical flood areas...")

any_flood = geometries[0]

for geom in geometries[1:]:
    any_flood = any_flood.union(geom)

if not any_flood.is_valid:
    print("Repairing final geometry...")
    any_flood = make_valid(any_flood)


# ============================================================
# KEEP POLYGONAL PARTS
# ============================================================

parts = gpd.GeoDataFrame(
    geometry=[any_flood],
    crs=TARGET_CRS
).explode(
    index_parts=False,
    ignore_index=True
)

parts = parts[
    parts.geometry.geom_type.isin(
        ["Polygon", "MultiPolygon"]
    )
].copy()

any_flood = parts.geometry.union_all()


# ============================================================
# SAVE
# ============================================================

final = gpd.GeoDataFrame(
    {
        "class": ["Any Historical Flood"],
        "events": ["2010 OR 2014 OR 2022"],
        "source": ["UNOSAT-derived inventories"],
    },
    geometry=[any_flood],
    crs=TARGET_CRS
)

final.to_file(
    OUTPUT,
    layer="any_flood",
    driver="GPKG"
)


# ============================================================
# EQUAL-AREA REPORT
# ============================================================

area_gdf = final.to_crs(AREA_CRS)

area_km2 = (
    area_gdf.geometry.area.iloc[0]
    / 1e6
)


print("\n" + "=" * 80)
print("ANY-HISTORICAL-FLOOD INVENTORY COMPLETE")
print("=" * 80)

print(
    f"Union flood area: "
    f"{area_km2:,.1f} km²"
)

print(
    f"Geometry: "
    f"{final.geometry.geom_type.iloc[0]}"
)

print(f"CRS: {final.crs}")

print(
    f"\nSaved:\n"
    f"{OUTPUT}"
)