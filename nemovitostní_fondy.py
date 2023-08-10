
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
import streamlit as st
from PIL import Image
import base64
import io

# Load the data
@st.cache_data
def load_data():
    df = pd.read_csv("FKI_fondy_streamlit.csv")
    return df

df = load_data()

df.rename(columns={'Rozložení portfolia':"Portfolio"},inplace=True)


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


import re

def dominant_category(text):
    # Vytvořte slovník s klíčovými slovy pro každou kategorii
    categories = {
        "kancelářské": ["kancelářské", "kancelář", "administrativní","office"],
        "výrobní": ["výrobní", "výroba"],
        "logistické": ["logistika", "logistické","logistika a výroba"],
        "obchodní": ["obchodní"],
        "retail": ["retail"],
        "rezidenční": ["rezidenční"],
        "průmysl/logistika": ["průmysl/logistika"]
    }
    
    # Pokud text není řetězec, vrať "Neznámé"
    if not isinstance(text, str):
        return "Neznámé"
    
    # Rozdělení řetězce na jednotlivé páry (procento, kategorie)
    pairs = re.findall(r'(\d+\.?\d* %) ([\w\s]+)', text)
    dominant_percentage = 0
    dominant_category = None
    
    # Pro každý pár extrakce procenta a identifikace kategorie
    for percentage, category in pairs:
        percentage = float(percentage.replace(' %', '').replace(',', '.'))
        for key, keywords in categories.items():
            if any(keyword in category for keyword in keywords):
                if percentage > dominant_percentage:
                    dominant_percentage = percentage
                    dominant_category = key
                    


    # Pokud dominantní kategorie nemá více než 50 %, vrať "Vyrovnané"
    if dominant_percentage <= 50:
        return "Vyrovnané"
    elif dominant_category:
        return f"Převažuje {dominant_category}"
    else:
        return "jiné"


df["Rozložení portfolia"] = df["Portfolio"].apply(dominant_category)




st.title("Fondy kvalifikovaných investorů")


def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Přidat filtrování")

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        # Skryjeme sloupec "Portfolio" v nabídce
        available_columns = [col for col in df.columns if col != "Portfolio"]
        to_filter_columns = st.multiselect("Filtrovat přehled podle:", available_columns)
        
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
    
        # Určení, zda je proměnná kategorická na základě nečíselných hodnot
            if df[column].apply(lambda x: not pd.api.types.is_number(x)).any():
                unique_values = df[column].dropna().unique()
                user_cat_input = right.multiselect(
                column,
                unique_values,
                default=list(unique_values)
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    column,
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    column,
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]

    return df



# Configure the image column
image_column = st.column_config.ImageColumn(label="Poskytovatel", width="medium")


df.set_index('Poskytovatel', inplace=True)

# Display the filtered data

filtered_df = filter_dataframe(df)

st.dataframe(filtered_df.drop(columns=["Rozložení portfolia"]), hide_index=True, column_config={"Poskytovatel": image_column}, height=428)
