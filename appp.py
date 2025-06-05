import streamlit as st
import ee
import geemap.foliumap as geemap
import requests
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
from tensorflow.keras.models import load_model
import joblib
import os
import datetime
import pandas as pd
import plotly.express as px

# Google API Key (hardcoded)
API_KEY = "AIzaSyCyFgbJ3ECkERB3AgDWJQF6qFlIA0B7gw8"  # Replace with your actual API key

# Define pixel-to-meter conversion scale (Adjust as per real-world data)
PIXEL_TO_METER_SCALE = 0.5

# Authenticate & Initialize Google Earth Engine
try:
    ee.Initialize()
except Exception as e:
    st.error(f"Google Earth Engine authentication failed: {str(e)}. Run earthengine authenticate.")

# Load ML models
@st.cache_resource
def load_models():
    try:
        # Use absolute paths that work for your system
        depth_model = load_model(r"C:\Users\USER\Documents\model\model_depth.h5", compile=False)
        yolo_model = YOLO(r"C:\Users\USER\Documents\model\runs\segment\train20\weights\best.pt").to("cpu")
        scaler = joblib.load(r"C:\Users\USER\Documents\model\scaler.pkl")
        encoder = joblib.load(r"C:\Users\USER\Documents\model\encoder.pkl")
        return depth_model, yolo_model, scaler, encoder
    except Exception as e:
        st.error(f"Error loading models: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None, None

# Function to fetch coordinates from lake name
@st.cache_data(ttl=3600)
def get_lake_coordinates(lake_name):
    try:
        search_url = f"https://maps.googleapis.com/maps/api/place/findplacefromtext/json?input={lake_name}&inputtype=textquery&fields=geometry,name&key={API_KEY}"
        response = requests.get(search_url).json()
        
        if "candidates" in response and response["candidates"]:
            location_data = response["candidates"][0]
            lat = location_data["geometry"]["location"]["lat"]
            lng = location_data["geometry"]["location"]["lng"]
            name = location_data.get("name", lake_name)
            return lat, lng, name
        return None, None, None
    except Exception as e:
        st.error(f"Error finding location: {e}")
        return None, None, None

# Function to fetch satellite image using Google Earth Engine
@st.cache_data(ttl=3600)
def get_satellite_image(latitude, longitude, year, buffer_radius=5000):
    try:
        # Define the region (point buffer to get an area)
        region = ee.Geometry.Point([longitude, latitude]).buffer(buffer_radius)  # Buffer radius in meters
        
        # Set date range for the specified year
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        
        # Fetch satellite image from Sentinel-2
        dataset = ee.ImageCollection("COPERNICUS/S2") \
            .filterBounds(region) \
            .filterDate(start_date, end_date) \
            .sort("CLOUD_COVERAGE_ASSESSMENT") \
            .first()
        
        # Check if image exists
        if dataset is None:
            return None, None
        
        # Visualize the image with RGB bands
        image = dataset.visualize(min=0, max=3000, bands=["B4", "B3", "B2"])  # RGB bands
        
        # Get the thumbnail URL
        url = image.getThumbURL({
            "region": region,
            "dimensions": "400x400",
            "format": "png"
        })
        
        # Fetch the image
        img_data = requests.get(url).content
        return Image.open(BytesIO(img_data)), dataset
    
    except Exception as e:
        st.error(f"Error fetching image for {year}: {e}")
        return None, None

# Function to get central color of image
def get_central_color(image_np, crop_ratio=0.05):
    """Extracts central 5% of the image and computes mean color."""
    h, w = image_np.shape[:2]
    crop_size = int(min(h, w) * crop_ratio)
    center_x, center_y = w // 2, h // 2
    x1, x2 = center_x - crop_size, center_x + crop_size
    y1, y2 = center_y - crop_size, center_y + crop_size
    central_patch = image_np[y1:y2, x1:x2]
    avg_color = np.mean(central_patch, axis=(0, 1))
    return avg_color

# Function to estimate depth
def estimate_depth(image_np):
    """Predict depth category using the trained model."""
    try:
        avg_color = get_central_color(image_np)
        avg_color_scaled = scaler.transform([avg_color])  # Standardize
        predictions = depth_model.predict(avg_color_scaled)  # Predict
        predicted_class = np.argmax(predictions)
        
        # Map the numeric class to the appropriate depth category with weight
        depth_mapping = {
            0: "Shallow",     # weight 1
            1: "Medium",      # weight 2
            2: "Very Deep",   # weight 3
            3: "Shallow-4",   # weight 4
            4: "Deep"         # weight 5
        }
        
        # If you're using encoder's inverse_transform to get the original label
        predicted_label_raw = encoder.inverse_transform([predicted_class])[0]
        
        # Depending on your encoder's output, either use raw label or map it
        # If encoder already gives correct labels, use:
        # return predicted_label_raw
        
        # Otherwise, map to our defined categories:
        return depth_mapping.get(predicted_class, "Unknown")
        
    except Exception as e:
        st.error(f"Error estimating depth: {e}")
        import traceback
        st.error(traceback.format_exc())
        return "Unknown"

# Function to segment image and calculate area
def segment_image(image):
    """Segment water body using YOLO model and calculate area."""
    try:
        results = yolo_model.predict(image, conf=0.45)
        img = np.array(image)
        
        for result in results:
            if hasattr(result, 'masks') and result.masks is not None:
                # For newer YOLO versions
                if hasattr(result.masks, 'xy'):
                    masks = result.masks.xy
                    mask_image = np.zeros_like(img)
                    
                    for mask in masks:
                        mask = np.array(mask, dtype=np.int32)
                        cv2.fillPoly(mask_image, [mask], color=(0, 0, 255))
                    
                    img_with_mask = cv2.addWeighted(img, 1, mask_image, 0.5, 0)
                    
                    # Calculate area: Count masked pixels and convert to meters
                    masked_pixels = np.count_nonzero(mask_image[:, :, 2])
                    area_meters = masked_pixels * PIXEL_TO_METER_SCALE
                    
                    return img_with_mask, area_meters
                # For older YOLO versions
                else:
                    mask_image = np.zeros_like(img)
                    for mask in result.masks:
                        mask_array = mask.cpu().numpy()
                        mask_binary = (mask_array > 0.5).astype(np.uint8)
                        mask_binary = cv2.resize(mask_binary, (img.shape[1], img.shape[0]))
                        mask_image[:, :, 2] += mask_binary * 255
                    
                    img_with_mask = cv2.addWeighted(img, 1, mask_image, 0.5, 0)
                    
                    masked_pixels = np.count_nonzero(mask_image[:, :, 2])
                    area_meters = masked_pixels * PIXEL_TO_METER_SCALE
                    
                    return img_with_mask, area_meters
        
        return img, 0
    except Exception as e:
        st.error(f"Error segmenting image: {e}")
        import traceback
        st.error(traceback.format_exc())
        return np.array(image), 0

# Function to create depth and area graphs
# Function to create depth and area graphs
def create_comparison_graphs(years_data):
    """Create comparative graphs for depth and area over years."""
    try:
        # Convert to DataFrame
        df = pd.DataFrame(years_data)
        
        # Create a mapping from categorical depth to actual depth in feet
        depth_to_feet = {
            "Shallow": 10,        # 10 feet
            "Medium": 30,         # 30 feet
            "Deep": 60,           # 60 feet
            "Very Deep": 100,     # 100 feet
            "Shallow-4": 15,      # 15 feet (alternative shallow class)
            "Unknown": 0          # Unknown depth
        }
        
        # Convert depth categories to numerical values in feet for plotting
        df["Depth_Feet"] = df["Depth"].map(lambda x: depth_to_feet.get(x, 0))
        
        # Create Area chart
        fig_area = px.line(
            df, 
            x="Year", 
            y="Area", 
            markers=True, 
            title="Water Body Area Over Years",
            labels={"Area": "Area (sq. meters)", "Year": "Year"},
            line_shape="linear"
        )
        fig_area.update_traces(line=dict(color="blue", width=2), marker=dict(size=8))
        fig_area.update_layout(
            xaxis=dict(tickmode="linear"),
            yaxis=dict(title="Area (sq. meters)"),
            hovermode="x unified"
        )
        
        # Create Depth chart with actual depth values
        fig_depth = px.line(
            df, 
            x="Year", 
            y="Depth_Feet", 
            markers=True, 
            title="Water Depth Over Years",
            labels={"Depth_Feet": "Depth (feet)", "Year": "Year"},
            line_shape="linear",
            custom_data=["Depth"]  # Include original category for hover info
        )
        
        # Customize the depth chart
        fig_depth.update_traces(
            line=dict(color="green", width=2), 
            marker=dict(size=8),
            hovertemplate="Year: %{x}<br>Depth: %{customdata[0]} (%{y} feet)<extra></extra>"
        )
        
        # Add more reference lines/markers for depth ranges
        depth_ranges = [
            {"name": "Shallow", "min": 0, "max": 20, "color": "rgba(173, 216, 230, 0.3)"},
            {"name": "Medium", "min": 20, "max": 50, "color": "rgba(135, 206, 250, 0.3)"},
            {"name": "Deep", "min": 50, "max": 80, "color": "rgba(0, 191, 255, 0.3)"},
            {"name": "Very Deep", "min": 80, "max": 120, "color": "rgba(0, 0, 139, 0.3)"}
        ]
        
        for range_data in depth_ranges:
            fig_depth.add_shape(
                type="rect",
                x0=df["Year"].min() - 0.5,
                x1=df["Year"].max() + 0.5,
                y0=range_data["min"],
                y1=range_data["max"],
                fillcolor=range_data["color"],
                line=dict(width=0),
                layer="below"
            )
            
        # Final layout adjustments for depth chart
        fig_depth.update_layout(
            xaxis=dict(tickmode="linear"),
            yaxis=dict(
                title="Depth (feet)",
                range=[0, 120]  # Set appropriate range for y-axis
            ),
            hovermode="closest"
        )
        
        return fig_area, fig_depth
    except Exception as e:
        st.error(f"Error creating graphs: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None

# Load models
depth_model, yolo_model, scaler, encoder = load_models()

# Streamlit UI
st.title("Lake Depth & Area Estimation Using Google Earth Engine")

# Sidebar for inputs
with st.sidebar:
    st.header("Location Parameters")
    
    input_type = st.radio("Input Type", ["Coordinates", "Lake Name", "Upload Image"])
    
    if input_type == "Coordinates":
        latitude = st.number_input("Latitude", value=37.7749, format="%.6f")
        longitude = st.number_input("Longitude", value=-122.4194, format="%.6f")
        location_name = st.text_input("Location Name (Optional)")
    elif input_type == "Lake Name":
        lake_name = st.text_input("Enter the name of the lake:")
    else:
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    
    if input_type != "Upload Image":
        st.header("Analysis Parameters")
        buffer_radius = st.slider("Buffer radius (meters)", 1000, 10000, 5000, 500)
        
        # Year selection
        current_year = datetime.datetime.now().year
        start_year = st.selectbox("Start Year", range(2015, current_year + 1), index=3)
        end_year = st.selectbox("End Year", range(2015, current_year + 1), index=current_year - 2015)
    
    analyze_button = st.button("Analyze")

# Main content area
if analyze_button:
    if input_type == "Upload Image":
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Segment image and calculate area
            st.subheader("Segmentation Result")
            segmented_img, area = segment_image(image_np)
            st.image(segmented_img, caption="Segmented Water Areas", use_column_width=True)
            st.write(f"Estimated Area: {area:.2f} square meters")
            
            # Estimate depth
            st.subheader("Estimated Depth")
            depth_category = estimate_depth(image_np)
            st.write(f"Predicted Depth Category: {depth_category}")
        else:
            st.error("Please upload an image first")
    else:
        # Get coordinates based on input type
        if input_type == "Lake Name":
            with st.spinner("Finding lake coordinates..."):
                latitude, longitude, found_name = get_lake_coordinates(lake_name)
                
                if latitude and longitude:
                    location_name = found_name
                    st.success(f"Found coordinates for {found_name}: ({latitude}, {longitude})")
                else:
                    st.error("Location not found! Check the name.")
        
        # If we have valid coordinates, proceed with analysis
        if 'latitude' in locals() and 'longitude' in locals() and latitude is not None:
            display_name = location_name if 'location_name' in locals() and location_name else f"Location ({latitude}, {longitude})"
            
            # Create tabs for different years
            years = range(start_year, end_year + 1)
            tabs = st.tabs([str(year) for year in years])
            
            # Store raw images for potential saving
            raw_images = {}
            
            # Store results for each year to use in graphs
            year_data = []
            
            # Process each year
            for i, year in enumerate(years):
                with tabs[i]:
                    with st.spinner(f"Fetching satellite imagery for {year}..."):
                        img_result = get_satellite_image(latitude, longitude, year, buffer_radius)
                        
                    if img_result and img_result[0]:
                        img, raw_image = img_result
                        raw_images[year] = raw_image
                        
                        # Convert to numpy array for processing
                        img_np = np.array(img)
                        
                        # Display original image
                        st.image(img, caption=f"Satellite Image of {display_name} ({year})")
                        
                        # Segment image and calculate area
                        with st.spinner("Segmenting water bodies..."):
                            segmented_img, area = segment_image(img_np)
                            
                        # Estimate depth
                        with st.spinner("Estimating depth..."):
                            depth_category = estimate_depth(img_np)
                        
                        # Store data for graphs
                        year_data.append({
                            "Year": year,
                            "Area": area,
                            "Depth": depth_category
                        })
                        
                        # Display results
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Estimated Depth", depth_category)
                        with col2:
                            st.metric("Estimated Area", f"{area:.2f} sq. meters")
                        
                        st.image(segmented_img, caption="Segmented Water Areas")
                    else:
                        st.error(f"No satellite image found for {year}")
                        # Add dummy data for the graph if we couldn't get real data
                        year_data.append({
                            "Year": year,
                            "Area": 0,
                            "Depth": "Unknown"
                        })
            
            # Display location on map
            st.subheader("Location Map")
            map_widget = geemap.Map(center=(latitude, longitude), zoom=12)
            map_widget.add_basemap("SATELLITE")
            map_widget.add_marker(location=(latitude, longitude), popup=display_name)
            map_widget.to_streamlit(height=400)
            
            # Create and display comparison graphs
            st.subheader("Historical Comparison")
            
            # If we have multiple years of data, create comparison graphs
            if len(year_data) > 1:
                fig_area, fig_depth = create_comparison_graphs(year_data)
                
                if fig_area and fig_depth:
                    st.plotly_chart(fig_area, use_container_width=True)
                    st.plotly_chart(fig_depth, use_container_width=True)
                    
                    # Export data as CSV
                    df = pd.DataFrame(year_data)
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download Data as CSV",
                        data=csv,
                        file_name=f"{display_name}_analysis.csv",
                        mime="text/csv"
                    )
            else:
                st.info("Select multiple years to see comparison graphs")
                
            # Option to save images
            if st.button("Save Images to Disk"):
                output_folder = f"satellite_images_{latitude}&{longitude}"
                
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                    
                for year in years:
                    with st.spinner(f"Saving image for {year}..."):
                        try:
                            # Define the region
                            region = ee.Geometry.Point([longitude, latitude]).buffer(buffer_radius)
                            
                            # Export the image
                            if year in raw_images and raw_images[year]:
                                filename = os.path.join(output_folder, f"earth_image_{year}.tif")
                                geemap.ee_export_image(raw_images[year], filename=filename, scale=10, region=region)
                                st.success(f"Saved image for {year}")
                            else:
                                st.warning(f"No image available for {year}")
                        except Exception as e:
                            st.error(f"Error saving image for {year}: {e}")
                
                st.success(f"All available images saved to '{output_folder}'")