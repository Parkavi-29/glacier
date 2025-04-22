import streamlit as st
import leafmap.foliumap as leafmap
import os

st.set_page_config(layout="wide")
st.title("🗺️ Glacier Mask Viewer with Time Slider")

# Folder containing yearly TIFF files
tiff_folder = "glacier_tiffs"  # Folder with GeoTIFFs like GlacierMask_2015.tif

years = list(range(2015, 2024))
selected_year = st.slider("Select Year", min_value=2015, max_value=2023, step=1)

# Construct the file path
tif_path = os.path.join(tiff_folder, f"GlacierMask_{selected_year}.tif")

# Display map
st.subheader(f"🧊 Glacier Mask for Year: {selected_year}")
m = leafmap.Map(center=[30.95, 79.05], zoom=10)
if os.path.exists(tif_path):
    m.add_raster(tif_path, layer_name=f"Glacier {selected_year}", colormap="Blues", opacity=0.6)
else:
    st.warning("GeoTIFF not found. Make sure you've exported it from GEE and placed it in the folder.")
m.to_streamlit(height=600)
