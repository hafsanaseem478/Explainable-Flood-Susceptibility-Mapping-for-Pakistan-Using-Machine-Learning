from pathlib import Path
import geopandas as gpd
import pandas as pd

GDB = Path("Data/inventory/FL20100802PAK.gdb")
TARGET_CRS = "EPSG:3395"

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


def build_union(classes):
    frames = []

    for layer in LAYERS:
        gdf = gpd.read_file(GDB, layer=layer)

        if "Water_Class" not in gdf.columns:
            continue

        gdf = gdf[
            gdf["Water_Class"].isin(classes)
            & gdf.geometry.notna()
            & ~gdf.geometry.is_empty
        ].copy()

        if len(gdf) == 0:
            continue

        gdf = gdf.to_crs(TARGET_CRS)

        frames.append(gdf[["geometry"]])

    combined = gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True),
        crs=TARGET_CRS
    )

    return combined.geometry.union_all()


# Strict observed flood water
strict = build_union([1])

# Broader flood / possible flood
broad = build_union([1, 2])

# Official cumulative layer
cum = gpd.read_file(
    GDB,
    layer="Cumulative_FloodExtent"
).to_crs(TARGET_CRS)

cum_union = cum.geometry.union_all()


def compare(name, geom):
    intersection = geom.intersection(cum_union)

    geom_area = geom.area
    cum_area = cum_union.area
    shared = intersection.area

    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    print(f"Candidate area proxy: {geom_area / 1e6:,.1f} km²")
    print(f"Shared with cumulative: {shared / 1e6:,.1f} km²")

    print(
        f"% candidate inside cumulative: "
        f"{100 * shared / geom_area:.1f}%"
    )

    print(
        f"% cumulative covered by candidate: "
        f"{100 * shared / cum_area:.1f}%"
    )


compare("STRICT — Water_Class 1 only", strict)
compare("BROAD — Water_Class 1 + 2", broad)