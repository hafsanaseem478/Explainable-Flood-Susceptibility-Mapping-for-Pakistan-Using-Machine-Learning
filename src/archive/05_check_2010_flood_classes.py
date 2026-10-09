from pathlib import Path
import geopandas as gpd

GDB = Path("Data/inventory/FL20100802PAK.gdb")

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

for layer in LAYERS:

    gdf = gpd.read_file(GDB, layer=layer)

    if len(gdf) == 0:
        print(f"{layer}: EMPTY")
        continue

    water_class = (
        gdf["Water_Class"].dropna().unique()
        if "Water_Class" in gdf.columns
        else []
    )

    dates = (
        gdf["Sensor_Date"].dropna().unique()
        if "Sensor_Date" in gdf.columns
        else []
    )

    print(
        f"{layer:<25} "
        f"n={len(gdf):>7,}  "
        f"Water_Class={water_class}  "
        f"Date={dates[:2]}"
    )