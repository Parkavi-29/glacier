# Version with online GeoTIFF from Google Drive
import streamlit as st
import leafmap.foliumap as leafmap

st.set_page_config(layout="wide")
st.title("🧊 Glacier Mask Viewer - Google Drive Overlay")

geotiff_url = "https://drive.google.com/uc?id=1vhmmtAYCCMyU8wV5NqZNbkNkNf9mWChi"

m = leafmap.Map(center=[30.95, 79.05], zoom=10)
m.add_raster(geotiff_url, layer_name="Glacier Mask (from Drive)", opacity=0.6)
m.to_streamlit(height=600)
