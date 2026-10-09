from pathlib import Path
import geopandas as gpd
import pandas as pd

GDB = Path("Data/inventory/FL20100802PAK.gdb")

FLOOD_LAYERS = [
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

TARGET_CRS = "EPSG:3395"

print("=" * 80)
print("2010 CUMULATIVE VS INDIVIDUAL FLOOD LAYERS")
print("=" * 80)

# Cumulative layer

cum = gpd.read_file(
    GDB,
    layer="Cumulative_FloodExtent"
).to_crs(TARGET_CRS)

cum = cum[
    cum.geometry.notna()
    & ~cum.geometry.is_empty
].copy()

cum_union = cum.geometry.union_all()

print(f"\nCumulative polygons: {len(cum):,}")


# Individual flood layers

frames = []

for layer in FLOOD_LAYERS:

    gdf = gpd.read_file(GDB, layer=layer)

    if len(gdf) == 0:
        print(f"{layer}: EMPTY")
        continue

    gdf = gdf[
        gdf.geometry.notna()
        & ~gdf.geometry.is_empty
    ].copy()

    gdf = gdf.to_crs(TARGET_CRS)

    frames.append(gdf[["geometry"]])

    print(f"{layer}: {len(gdf):,} polygons")

combined = gpd.GeoDataFrame(
    pd.concat(frames, ignore_index=True),
    crs=TARGET_CRS
)

individual_union = combined.geometry.union_all()

print(f"\nIndividual flood polygons total: {len(combined):,}")


# Compare spatial overlap

intersection = cum_union.intersection(individual_union)

cum_area = cum_union.area
ind_area = individual_union.area
intersection_area = intersection.area

print("\n" + "=" * 80)
print("OVERLAP CHECK")
print("=" * 80)

print(
    f"Cumulative area proxy: "
    f"{cum_area / 1e6:,.1f} km²"
)

print(
    f"Individual-union area proxy: "
    f"{ind_area / 1e6:,.1f} km²"
)

print(
    f"Shared area proxy: "
    f"{intersection_area / 1e6:,.1f} km²"
)

print(
    f"\n% of cumulative covered by individual flood layers: "
    f"{100 * intersection_area / cum_area:.1f}%"
)

print(
    f"% of individual flood union inside cumulative layer: "
    f"{100 * intersection_area / ind_area:.1f}%"
)

print("\nDone.")