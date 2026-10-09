from pathlib import Path
import pyogrio

GDB = Path("Data/inventory/FL20100802PAK.gdb")

df = pyogrio.read_dataframe(
    GDB,
    layer="Water_Class",
    read_geometry=False
)

print(df.to_string(index=False))