import streamlit as st

#from my_library import *
import zipfile

from PIL import Image
import numpy as np
import cv2

import tkinter as tk
from tkinter import filedialog

import shutil
import os, io
import xml.etree.ElementTree as ET
from glob import glob
import pandas as pd
from shutil import copyfile
from matplotlib import patches
import numpy as np
import torch
import base64
from PIL import Image


# Import for API calls
import requests

# Import for navbar
from streamlit_option_menu import option_menu

# Import for dynamic tagging
from streamlit_tags import st_tags, st_tags_sidebar

# Imports for aggrid
import st_aggrid
from st_aggrid import AgGrid
from st_aggrid import GridOptionsBuilder
from st_aggrid.shared import JsCode
from st_aggrid import GridUpdateMode, DataReturnMode


# Add a download button for the dataframe
def download_button(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="data.csv">Download CSV file</a>'
    return href


# The code below is to control the layout width of the app.
if "widen" not in st.session_state:
    layout = "centered"
else:
    layout = "wide" if st.session_state.widen else "centered"
    

st.set_page_config(layout=layout, page_title="Image Analysis", page_icon="🤗")

# Set up session state so app interactions don't reset the app.
if not "valid_inputs_received" in st.session_state:
    st.session_state["valid_inputs_received"] = False

    #######################################################

# The block of code below is to display the title, logos and introduce the app.

c1, c2 = st.columns([0.4, 2])

with c1:

    st.image(
        "miscroscope.png",
        width=110,
    )

with c2:

    st.caption("")
    st.title("Image Analysis Pipeline: From Image to Personalised Reports")


st.sidebar.image(
    "30days_logo.png",
)

st.write("")

st.markdown(
    """
Goals: \n
Allow BioTech researchers to \n
✅  Generate personalisable reports to analyze microscopy images \n
✅  Increase number of observations: perform more experiments in less time \n
✅  Perform precise time-lapsed analysis of their experiments \n
\n 
Usage:\n
1️⃣ Select image (png) that you would like to analyze \n
2️⃣ Select what type of report you would like to generate \n
3️⃣ Observe results and download report \n
"""
)

st.write("")

st.sidebar.write("")


#######################################################

# The block of code below is to display information about Streamlit.

st.sidebar.markdown("---")

# Sidebar
st.sidebar.header("About")

st.sidebar.markdown(
    """
Web App created by Jeanne Chaverot for Scientists in the Biotech industry working with microscopic data.

If this product interests you please reach out to explore how we can adapt this technology to your data.

This template model is based on the YOLO Architecture trained on BCC public dataset.
"""
)



st.sidebar.header("Resources")
st.sidebar.markdown(
    """
- [GitHub](https://github.com/jeannechaverot)
- [LinkedIn](https://linkedin.com/in/jeanne-chaverot)
"""
)



#######################################################

#######################################################



# Set up Streamlit app
st.title('Image Processing and Excel Output')

# Get user input (folder)

uploaded_file = st.file_uploader("Upload your image", type=["jpg"], accept_multiple_files=False, key="imageUploader")

mapping_classes =  {0: "Platelets", 1: "RBC", 2: "WBC"}

# If a file has been uploaded, extract it to a temporary directory and process it
if uploaded_file is not None:

    # Add a title for the report type options
    st.subheader("Report type")
    
    # Selectbox for report type options
    report_type = st.selectbox("Select the report type", ["Raw Data", "Entity Counter", "Size Compute"])
    
    # Use the selected report type in your Streamlit application
    st.write("You selected:", report_type)


    file_contents = uploaded_file.read()
    np_array = np.frombuffer(file_contents, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    st.write("Succesfully imported the following image:")
    st.image(image, channels="BGR")


        # Model
    model_path = "yolov5_trained_cells.pt"
    model = torch.hub.load('yolov5', 'custom', source='local', path=model_path)

    result = model([image], size=640)

    result.save("blabla.png")

    tensor = result.xyxy
    array = tensor[0].numpy()
    raw_df = pd.DataFrame(array, columns=['x1', 'y1', 'x2', 'y2', 'confidence', 'class'])
    bounding_boxes = raw_df[['x1', 'y1', 'x2', 'y2']].values

    raw_df['class'] = raw_df['class'].apply(lambda x: mapping_classes[x])
    count_df = raw_df.groupby('class').count()
    count_df = pd.DataFrame(count_df.rename(columns={'confidence': 'count'})['count'])
    sizes_df = raw_df.copy()
    sizes_df['bounding box area (pixels)'] = (raw_df['x2']-raw_df['x1'])*(raw_df['y2']-raw_df['y1'])
    sizes_df = sizes_df[['class', 'bounding box area (pixels)']]
    # Convert the cv2 image to PIL format
    st.subheader("Detected organisms")
    image_pil = Image.fromarray(image)

    # Display the image with bounding boxes in Streamlit
    st.image(image_pil, channels="BGR")

    if report_type == "Raw Data":

        st.subheader("Raw report from detected organisms")
        st.dataframe(raw_df)
        st.markdown(download_button(raw_df), unsafe_allow_html=True)

    elif report_type == "Entity Counter":

        st.subheader("Counter report from detected organisms")
        st.dataframe(count_df)
        st.markdown(download_button(count_df), unsafe_allow_html=True)
    else:
        st.subheader("Size report from detected organisms")
        st.dataframe(sizes_df)
        st.markdown(download_button(sizes_df), unsafe_allow_html=True)
       




                                        
    # Walk through the directory and select the documents according to the filtering function
        
                                    #st.image(output_image)

    

    ## generate reports based on selected features



