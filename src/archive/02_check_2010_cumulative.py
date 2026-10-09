from pathlib import Path
import geopandas as gpd

GDB = Path("Data/inventory/FL20100802PAK.gdb")
LAYER = "Cumulative_FloodExtent"

print("=" * 80)
print("2010 CUMULATIVE FLOOD INVENTORY CHECK")
print("=" * 80)

gdf = gpd.read_file(GDB, layer=LAYER)

print(f"\nFeatures: {len(gdf):,}")
print(f"CRS: {gdf.crs}")
print(f"Geometry: {gdf.geometry.geom_type.value_counts().to_dict()}")
print(f"Bounds: {gdf.total_bounds}")

for col in [
    "Water_Class",
    "Water_StatusID",
    "Confidence_ID",
    "Sensor_ID",
    "Sensor_Date",
    "Field_Validation",
]:
    if col in gdf.columns:

        values = gdf[col].dropna().unique()

        print(f"\n{col}:")
        print(values[:50])

print("\nDone.")