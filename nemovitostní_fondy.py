



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

# Definujeme responzivní CSS
styles = """
    <style>
        /* Pro obrazovky menší než 600px */
        @media (max-width: 400px) {
            .filter-group {
                flex-direction: column !important;
            }
        }
    </style>
"""

# Vložíme styly do Streamlitu
st.markdown(styles, unsafe_allow_html=True)

# Vytvoření 2 sloupců pro filtry (můžete si upravit počet podle potřeby)
with st.container():
    col1, col2 = st.columns(2)

    # Filtr pro Výnos 2022 (v %) v prvním sloupci
    with col1:
        if st.checkbox(label="Výnos 2022",value = False, key="checkbox_2022"):
            ranges = ["Více než 10 %","5 % - 10 %", "0 % - 5 %","Menší než 0 %"]
            selected_range_2022 = st.selectbox("Výnos 2022 (v %)", ranges)
        
            #  Přesuňte logiku filtru dovnitř podmínky zaškrtávacího políčka
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


# Configure the image column
image_column = st.column_config.ImageColumn(label="Poskytovatel", width="medium")

filtered_data.set_index('Poskytovatel', inplace=True)

# Display the filtered data
st.dataframe(filtered_data, column_config={"Poskytovatel": image_column}, height=428)

