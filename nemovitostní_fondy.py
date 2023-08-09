



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


# Initialize filtered_data as original data
filtered_data = df

# Sidebar filters
st.header("Filtry")

# Vytvoření 7 sloupců pro filtry (můžete si upravit počet podle potřeby)
col1, col2, col3, col4, col5 = st.columns(5)

# Filtr pro Výnos 2022 (v %) v prvním sloupci
with col1:
    if st.checkbox(label="Výnos 2022",value = False, key="checkbox_2022"):
        ranges = ["Více než 10 %","5 % - 10 %", "0 % - 5 %","Menší než 0 %"]
        selected_range_2022 = st.selectbox("Výnos 2022 (v %)", ranges)
        
        # Přesuňte logiku filtru dovnitř podmínky zaškrtávacího políčka
        if "Menší než 0 %" == selected_range_2022:
            filtered_data = filtered_data[filtered_data["Výnos 2022 (v %)"] < 0]
        
        elif "0 % - 5 %" == selected_range_2022:
            filtered_data = filtered_data[(filtered_data["Výnos 2022 (v %)"] >= 0) & (filtered_data["Výnos 2022 (v %)"] <= 5)]
        
        elif "5 % - 10 %" == selected_range_2022:
            filtered_data = filtered_data[(filtered_data["Výnos 2022 (v %)"] > 5) & (filtered_data["Výnos 2022 (v %)"] <= 10)]
        
        elif "Více než 10 %" == selected_range_2022:
            filtered_data = filtered_data[filtered_data["Výnos 2022 (v %)"] > 10]



# Filter for Výnos 2021 (v %)
with col2:
    if st.checkbox(label="Výnos 2021",value = False, key="checkbox_2021"):
        ranges_2021 = ["Více než 10 %","5 % - 10 %", "0 % - 5 %","Menší než 0 %"]
        selected_range_2021 = st.selectbox("Výnos 2021 (v %)", ranges_2021)
        
        # Přesuňte logiku filtru dovnitř podmínky zaškrtávacího políčka
        if "Menší než 0 %" == selected_range_2021:
            filtered_data = filtered_data[filtered_data["Výnos 2021 (v %)"] < 0]
    
        elif "0 % - 5 %" == selected_range_2021:
            filtered_data = filtered_data[(filtered_data["Výnos 2021 (v %)"] >= 0) & (filtered_data["Výnos 2021 (v %)"] <= 5)]
        
        elif "5 % - 10 %" == selected_range_2021:
            filtered_data = filtered_data[(filtered_data["Výnos 2021 (v %)"] > 5) & (filtered_data["Výnos 2021 (v %)"] <= 10)]
        
        elif "Více než 10 %" == selected_range_2021:
            filtered_data = filtered_data[filtered_data["Výnos 2021 (v %)"] > 10]


# Filtr pro Výnos 2020 (v %) v prvním sloupci
with col3:
    if st.checkbox(label="Výnos 2020",value = False, key="checkbox_2020"):
        ranges = ["Více než 10 %","5 % - 10 %", "0 % - 5 %","Menší než 0 %"]
        selected_range_2020 = st.selectbox("Výnos 2020 (v %)", ranges)
        
        # Přesuňte logiku filtru dovnitř podmínky zaškrtávacího políčka
        if "Menší než 0 %" == selected_range_2020:
            filtered_data = filtered_data[filtered_data["Výnos 2020 (v %)"] < 0]
        
        elif "0 % - 5 %" == selected_range_2020:
            filtered_data = filtered_data[(filtered_data["Výnos 2020 (v %)"] >= 0) & (filtered_data["Výnos 2020 (v %)"] <= 5)]
        
        elif "5 % - 10 %" == selected_range_2020:
            filtered_data = filtered_data[(filtered_data["Výnos 2020 (v %)"] > 5) & (filtered_data["Výnos 2020 (v %)"] <= 10)]
        
        elif "Více než 10 %" == selected_range_2020:
            filtered_data = filtered_data[filtered_data["Výnos 2020 (v %)"] > 10]



with col4:
    if st.checkbox(label="Výnos od založení",value = False, key="checkbox_vynos_od_zalozeni"):
        ranges_zalozeni = ["Více než 10 %","5 % - 10 %", "0 % - 5 %","Menší než 0 %"]
        selected_range_zalozeni = st.selectbox("Výnos od založení (% p.a.)", ranges_zalozeni)
        
        # Přesuňte logiku filtru dovnitř podmínky zaškrtávacího políčka
        if "Menší než 0 %" == selected_range_zalozeni:
            filtered_data = filtered_data[filtered_data["Výnos od založení (% p.a.)"] < 0]
        
        elif "0 % - 5 %" == selected_range_zalozeni:
            filtered_data = filtered_data[(filtered_data["Výnos od založení (% p.a.)"] >= 0) & (filtered_data["Výnos od založení (% p.a.)"] <= 5)]
        
        elif "5 % - 10 %" == selected_range_zalozeni:
            filtered_data = filtered_data[(filtered_data["Výnos od založení (% p.a.)"] > 5) & (filtered_data["Výnos od založení (% p.a.)"] <= 10)]
        
        elif "Více než 10 %" == selected_range_zalozeni:
            filtered_data = filtered_data[filtered_data["Výnos od založení (% p.a.)"] > 10]


# Filter for TER (v %)
with col5:
    if st.checkbox(label="TER",value = False, key="checkbox_TER"):
        min_ter = st.number_input("Min TER (v %)", float(df["TER (v %)"].min()))
        max_ter = st.number_input("Max TER (v %)", float(df["TER (v %)"].max()))
        filtered_data = filtered_data[filtered_data["TER (v %)"].between(min_ter, max_ter)]


col6, col7, col8, col9, col10 = st.columns(5)

# Filter for LTV (v %)
with col6:
    if st.checkbox(label="LTV",value = False, key="checkbox_LTV"):
        min_ltv = st.number_input("Min LTV (v %)", float(df["LTV (v %)"].min()))
        max_ltv = st.number_input("Max LTV (v %)", float(df["LTV (v %)"].max()))
        filtered_data = filtered_data[filtered_data["LTV (v %)"].between(min_ltv, max_ltv)]

# Filter for WAULT
with col7:
    if st.checkbox("WAULT", False, key="checkbox_WAULT"):
        min_wault = st.number_input("Min WAULT", float(df["WAULT"].min()))
        max_wault = st.number_input("Max WAULT", float(df["WAULT"].max()))
        filtered_data = filtered_data[filtered_data["WAULT"].between(min_wault, max_wault)]

# Filter for YIELD (v %)
with col8:
    if st.checkbox(label="YIELD",value = False, key="checkbox_YIELD"):
        min_yield = st.number_input("Min YIELD (v %)", float(df["YIELD (v %)"].min()))
        max_yield = st.number_input("Max YIELD (v %)", float(df["YIELD (v %)"].max()))
        filtered_data = filtered_data[filtered_data["YIELD (v %)"].between(min_yield, max_yield)]

# Filter for NAV (v mld. Kč)
with col9:
    if st.checkbox(label="NAV",value = False, key="checkbox_NAV"):
        min_nav = st.number_input("Min NAV (v mld. Kč)", float(df["NAV (v mld. Kč)"].min()))
        max_nav = st.number_input("Max NAV (v mld. Kč)", float(df["NAV (v mld. Kč)"].max()))
        filtered_data = filtered_data[filtered_data["NAV (v mld. Kč)"].between(min_nav, max_nav)]

# Filter for Počet nemovitostí
with col10:
    if st.checkbox("Počet nemovitostí", False, key="checkbox_pocet_nemovitosti"):
        ranges_2020 = ["Více než 20","10 - 20", "0 - 10"]
        selected_range_2020 = st.selectbox("Počet nemovitostí", ranges_2020)
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

