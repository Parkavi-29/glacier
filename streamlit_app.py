import streamlit as st
import leafmap.foliumap as leafmap
import os

# Streamlit UI setup
st.set_page_config(layout="wide")
st.title("🧊 Himalayan Glacier Melt Visualizer")
st.markdown("Use the slider to explore glacier mask overlays from 2015 to 2023.")

# Folder containing exported GeoTIFFs
tiff_folder = "glacier_tiffs"

# Time slider for year selection
years = list(range(2015, 2024))
selected_year = st.slider("Select Year", min_value=2015, max_value=2023, step=1)

# Construct file path
tif_path = os.path.join(tiff_folder, f"GlacierMask_{selected_year}.tif")

# Initialize map
m = leafmap.Map(center=[30.95, 79.05], zoom=10)

# Add glacier mask if available
if os.path.exists(tif_path):
    m.add_raster(tif_path, layer_name=f"Glacier Mask {selected_year}", colormap="Blues", opacity=0.6)
    st.success(f"Displaying Glacier Mask for {selected_year}")
else:
    st.warning("GeoTIFF not found. Please make sure it's in the 'glacier_tiffs/' folder.")

# Show map
m.to_streamlit(height=600)
