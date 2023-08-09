



import streamlit as st
import pandas as pd
from PIL import Image
import base64
import io

# Load the data
@st.cache_data
def load_data():
    df = pd.read_csv("FKI_fondy_streamlit.csv")
    return df

df = load_data()

# Convert image to Base64
def image_to_base64(img_path, output_size=(441, 100)):
    # Open an image file
    with Image.open(img_path) as img:
        # Resize image
        img = img.resize(output_size)
        # Convert image to PNG and then to base64
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"
    

# Apply conversion function to the column with image paths
df["Poskytovatel"] = df["Poskytovatel"].apply(image_to_base64)


# Sidebar filters
st.sidebar.header("Filtry")

# Initialize filtered_data as original data
filtered_data = df

# Filter for Výnos 2022 (v %)

if st.sidebar.checkbox("Výnos 2022 (v %)", False, key="checkbox_2022"):
    ranges = ["Více než 10 %","5 % - 10 %", "0 % - 5 %","Menší než 0 %"]
    selected_range = st.sidebar.selectbox("Výnos 2022 (v %)", ranges)

    if "Menší než 0 %" == selected_range:
        filtered_data = filtered_data[filtered_data["Výnos 2022 (v %)"] < 0]
    
    elif "0 % - 5 %" == selected_range:
        filtered_data = filtered_data[(filtered_data["Výnos 2022 (v %)"] >= 0) & (filtered_data["Výnos 2022 (v %)"] <= 5)]
        
    elif "5 % - 10 %" == selected_range:
        filtered_data = filtered_data[(filtered_data["Výnos 2022 (v %)"] > 5) & (filtered_data["Výnos 2022 (v %)"] <= 10)]
        
    elif "Více než 10 %" == selected_range:
        filtered_data = filtered_data[filtered_data["Výnos 2022 (v %)"] > 10]



# Filter for Výnos 2021 (v %)
if st.sidebar.checkbox("Výnos 2021 (v %)", False, key="checkbox_2021"):
    ranges_2021 = ["Více než 10 %","5 % - 10 %", "0 % - 5 %","Menší než 0 %"]
    selected_range_2021 = st.sidebar.selectbox("Výnos 2021 (v %)", ranges_2021)

    if "Menší než 0 %" == selected_range_2021:
        filtered_data = filtered_data[filtered_data["Výnos 2021 (v %)"] < 0]
    
    elif "0 % - 5 %" == selected_range_2021:
        filtered_data = filtered_data[(filtered_data["Výnos 2021 (v %)"] >= 0) & (filtered_data["Výnos 2021 (v %)"] <= 5)]
        
    elif "5 % - 10 %" == selected_range_2021:
        filtered_data = filtered_data[(filtered_data["Výnos 2021 (v %)"] > 5) & (filtered_data["Výnos 2021 (v %)"] <= 10)]
        
    elif "Více než 10 %" == selected_range_2021:
        filtered_data = filtered_data[filtered_data["Výnos 2021 (v %)"] > 10]


# Filter for Výnos 2020 (v %)
if st.sidebar.checkbox("Výnos 2020 (v %)", False, key="checkbox_2020"):
    ranges_2020 = ["Více než 10 %","5 % - 10 %", "0 % - 5 %","Menší než 0 %"]
    selected_range_2020 = st.sidebar.selectbox("Výnos 2020 (v %)", ranges_2020)

    if "Menší než 0 %" == selected_range_2020:
        filtered_data = filtered_data[filtered_data["Výnos 2020 (v %)"] < 0]
    
    elif "0 % - 5 %" == selected_range_2020:
        filtered_data = filtered_data[(filtered_data["Výnos 2020 (v %)"] >= 0) & (filtered_data["Výnos 2020 (v %)"] <= 5)]
        
    elif "5 % - 10 %" == selected_range_2020:
        filtered_data = filtered_data[(filtered_data["Výnos 2020 (v %)"] > 5) & (filtered_data["Výnos 2020 (v %)"] <= 10)]
        
    elif "Více než 10 %" == selected_range_2020:
        filtered_data = filtered_data[filtered_data["Výnos 2020 (v %)"] > 10]


# Filter for Výnos od založení (% p.a.)
if st.sidebar.checkbox("Výnos od založení (% p.a.)", False, key="checkbox_vynos_od_zalozeni"):
    ranges_zalozeni = ["Více než 10 %","5 % - 10 %", "0 % - 5 %","Menší než 0 %"]
    selected_range_2020 = st.sidebar.selectbox("Výnos od založení (% p.a.)", ranges_zalozeni)

    if "Menší než 0 %" == selected_range_2020:
        filtered_data = filtered_data[filtered_data["Výnos od založení (% p.a.)"] < 0]
    
    elif "0 % - 5 %" == selected_range_2020:
        filtered_data = filtered_data[(filtered_data["Výnos od založení (% p.a.)"] >= 0) & (filtered_data["Výnos od založení (% p.a.)"] <= 5)]
        
    elif "5 % - 10 %" == selected_range_2020:
        filtered_data = filtered_data[(filtered_data["Výnos od založení (% p.a.)"] > 5) & (filtered_data["Výnos od založení (% p.a.)"] <= 10)]
        
    elif "Více než 10 %" == selected_range_2020:
        filtered_data = filtered_data[filtered_data["Výnos od založení (% p.a.)"] > 10]


# Filter for TER (v %)
if st.sidebar.checkbox("TER (v %)", False, key="checkbox_TER"):
    min_2022 = st.sidebar.number_input("Min TER (v %)", float(df["TER (v %)"].min()))
    max_2022 = st.sidebar.number_input("Max TER (v %)", float(df["TER (v %)"].max()))
    filtered_data = filtered_data[filtered_data["TER (v %)"].between(min_2022, max_2022)]

# Filter for LTV (v %)
if st.sidebar.checkbox("LTV (v %)", False, key="checkbox_LTV"):
    min_2022 = st.sidebar.number_input("Min LTV (v %)", float(df["LTV (v %)"].min()))
    max_2022 = st.sidebar.number_input("Max LTV (v %)", float(df["LTV (v %)"].max()))
    filtered_data = filtered_data[filtered_data["LTV (v %)"].between(min_2022, max_2022)]

# Filter for WAULT
if st.sidebar.checkbox("WAULT", False, key="checkbox_WAULT"):
    min_2022 = st.sidebar.number_input("Min WAULT", float(df["WAULT"].min()))
    max_2022 = st.sidebar.number_input("Max WAULT", float(df["WAULT"].max()))
    filtered_data = filtered_data[filtered_data["WAULT"].between(min_2022, max_2022)]

# Filter for YIELD (v %)
if st.sidebar.checkbox("YIELD (v %)", False, key="checkbox_YIELD"):
    min_2022 = st.sidebar.number_input("Min YIELD (v %)", float(df["YIELD (v %)"].min()))
    max_2022 = st.sidebar.number_input("Max YIELD (v %)", float(df["YIELD (v %)"].max()))
    filtered_data = filtered_data[filtered_data["YIELD (v %)"].between(min_2022, max_2022)]


# Filter for NAV (v mld. Kč)
if st.sidebar.checkbox("NAV (v mld. Kč)", False, key="checkbox_NAV"):
    min_2022 = st.sidebar.number_input("Min NAV (v mld. Kč)", float(df["NAV (v mld. Kč)"].min()))
    max_2022 = st.sidebar.number_input("Max NAV (v mld. Kč)", float(df["NAV (v mld. Kč)"].max()))
    filtered_data = filtered_data[filtered_data["NAV (v mld. Kč)"].between(min_2022, max_2022)]

# Filter for Počet nemovitostí
if st.sidebar.checkbox("Počet nemovitostí", False, key="checkbox_pocet_nemovitosti"):
    ranges_2020 = ["Více než 20","10 - 20", "0 - 10"]
    selected_range_2020 = st.sidebar.selectbox("Počet nemovitostí", ranges_2020)
    
    if "0 - 10" == selected_range_2020:
        filtered_data = filtered_data[(filtered_data["Počet nemovitostí"] >= 0) & (filtered_data["Počet nemovitostí"] <= 10)]
        
    elif "10 - 20" == selected_range_2020:
        filtered_data = filtered_data[(filtered_data["Počet nemovitostí"] > 10) & (filtered_data["Počet nemovitostí"] <= 20)]
        
    elif "Více než 20" == selected_range_2020:
        filtered_data = filtered_data[filtered_data["Počet nemovitostí"] > 20]


# Configure the image column
image_column = st.column_config.ImageColumn(label="Poskytovatel", width="medium")


filtered_data.set_index('Poskytovatel', inplace=True)

# Display the filtered data
st.dataframe(filtered_data, column_config={"Poskytovatel": image_column}, height=428)


