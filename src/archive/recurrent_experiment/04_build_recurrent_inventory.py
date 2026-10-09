from pathlib import Path

import geopandas as gpd
from shapely import make_valid


# PATHS

BASE = Path("Data/inventory/processed")

FILE_2010 = BASE / "flood_extent_2010.gpkg"
FILE_2014 = BASE / "flood_extent_2014.gpkg"
FILE_2022 = BASE / "flood_extent_2022.gpkg"

OUTPUT = BASE / "recurrent_flood_2010_2014_2022.gpkg"

TARGET_CRS = "EPSG:3395"
AREA_CRS = "EPSG:6933"   # equal-area CRS for reporting areas


print("=" * 80)
print("BUILDING RECURRENT FLOOD INVENTORY")
print("=" * 80)


# READ THREE CLEAN EVENT INVENTORIES

g2010 = gpd.read_file(FILE_2010).to_crs(TARGET_CRS)
g2014 = gpd.read_file(FILE_2014).to_crs(TARGET_CRS)
g2022 = gpd.read_file(FILE_2022).to_crs(TARGET_CRS)

geom2010 = g2010.geometry.union_all()
geom2014 = g2014.geometry.union_all()
geom2022 = g2022.geometry.union_all()

print("\nLoaded:")
print("✓ 2010")
print("✓ 2014")
print("✓ 2022")


# INTERSECTION = AREAS FLOODED IN ALL THREE EVENT INVENTORIES

print("\nIntersecting 2010, 2014 and 2022...")

common = (
    geom2010
    .intersection(geom2014)
    .intersection(geom2022)
)

if common.is_empty:
    raise RuntimeError(
        "The three flood inventories have no common area."
    )

if not common.is_valid:
    print("Repairing recurrent geometry...")
    common = make_valid(common)


# KEEP POLYGONAL GEOMETRY ONLY

parts = gpd.GeoDataFrame(
    geometry=[common],
    crs=TARGET_CRS
).explode(index_parts=False)

parts = parts[
    parts.geometry.geom_type.isin(
        ["Polygon", "MultiPolygon"]
    )
].copy()

if len(parts) == 0:
    raise RuntimeError(
        "Intersection exists but contains no polygon geometry."
    )

common = parts.geometry.union_all()


# FINAL OUTPUT

final = gpd.GeoDataFrame(
    {
        "class": ["Recurrent Flood"],
        "events": ["2010 + 2014 + 2022"],
        "source": ["UNOSAT-derived inventories"],
    },
    geometry=[common],
    crs=TARGET_CRS
)

final.to_file(
    OUTPUT,
    layer="recurrent_flood",
    driver="GPKG"
)


# AREA CHECK IN EQUAL-AREA CRS

def area_km2(geometry):
    temp = gpd.GeoDataFrame(
        geometry=[geometry],
        crs=TARGET_CRS
    ).to_crs(AREA_CRS)

    return temp.geometry.area.iloc[0] / 1e6


a2010 = area_km2(geom2010)
a2014 = area_km2(geom2014)
a2022 = area_km2(geom2022)
acommon = area_km2(common)


# REPORT

print("\n" + "=" * 80)
print("RECURRENT FLOOD INVENTORY COMPLETE")
print("=" * 80)

print(f"2010 flood area:   {a2010:,.1f} km²")
print(f"2014 flood area:   {a2014:,.1f} km²")
print(f"2022 flood area:   {a2022:,.1f} km²")

print(
    f"\nCommon recurrent area: "
    f"{acommon:,.1f} km²"
)

print(
    f"Common as % of 2010: "
    f"{100 * acommon / a2010:.2f}%"
)

print(
    f"Common as % of 2014: "
    f"{100 * acommon / a2014:.2f}%"
)

print(
    f"Common as % of 2022: "
    f"{100 * acommon / a2022:.2f}%"
)

print(f"\nGeometry: {final.geometry.geom_type.iloc[0]}")
print(f"CRS: {final.crs}")

print(f"\nSaved:\n{OUTPUT}")