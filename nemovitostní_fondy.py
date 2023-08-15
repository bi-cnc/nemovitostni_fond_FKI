
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

st.set_page_config(layout="wide")

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


# Nahraďte NaN hodnoty "Neuvedeno"
df["Cílený roční výnos"].fillna("Neuvedeno", inplace=True)

def convert_yield_to_float(yield_value):
    if yield_value == "Neuvedeno":
        return -1
    if isinstance(yield_value, str):
        # Pokud obsahuje rozsah, vytvoříme kombinovanou hodnotu
        if '-' in yield_value:
            first_val, second_val = map(lambda x: float(x.replace('%', '').strip()), yield_value.split('-'))
            # Vracíme kombinovanou hodnotu
            return first_val + second_val * 0.01
        # Odeberte procenta a převeďte na float
        yield_value = yield_value.replace('%', '').replace(',', '.').strip()
        # Pokud obsahuje '+', přidáme malou hodnotu pro řazení
        if '+' in yield_value:
            yield_value = yield_value.replace('+', '').strip()
            return float(yield_value) + 0.001  # přidáme 0.001 pro řazení
        else:
            return float(yield_value)
    return None


def extract_number_from_string(s):
    numbers = re.findall(r"(\d+)", s)
    if numbers:
        return int(numbers[0])
    return 0



# Zbytek kódu zůstává stejný

# Seřazení hodnot ve sloupci "Cílený roční výnos"
sorted_yield_values = sorted(df["Cílený roční výnos"].unique(), key=convert_yield_to_float)


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


def convert_fee_to_float_simple(fee_value):
    if isinstance(fee_value, str):
        # Ořízne řetězec na základě první závorky (pokud existuje)
        fee_value = fee_value.split('(')[0].strip()

        # Zkusíme extrahovat čísla z řetězce
        numbers = re.findall(r"(\d+\.?\d*)", fee_value)
        if not numbers:  # pokud nejsou žádná čísla, vrátíme None
            return None

        if '%' in fee_value:
            # Pokud obsahuje více částí oddělených čárkami, vezmeme první část
            fee_value = fee_value.split(',')[0].strip()
            
            # Pokud obsahuje rozsah, vytvoříme kombinovanou hodnotu
            if '-' in fee_value:
                fee_parts = fee_value.split('-')
                # Vezmeme první číslo z rozsahu
                return float(fee_parts[0].replace('%', '').strip())
            
            # Extrakce čísla ze stringu
            fee_value = numbers[0]
            return float(fee_value)
    return None





fee_columns = ["Vstupní poplatek", "Manažerský poplatek", "Výkonnostní poplatek", "Výstupní poplatek"]


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

            if column == "Rozložení portfolia":
                unique_portfolio_values = df[column].dropna().unique()
                user_portfolio_input = right.multiselect(
                "Rozložení portfolia",
                unique_portfolio_values,
                default=list(unique_portfolio_values)
                )
                df = df[df[column].isin(user_portfolio_input)]
                continue
            
            if column == "Cílený roční výnos":
                user_yield_input = right.multiselect(
                    "Cílený roční výnos",
                    sorted_yield_values,
                    default=sorted_yield_values  # ve výchozím stavu označit všechny hodnoty
                )
                df = df[df["Cílený roční výnos"].isin(user_yield_input)]
                continue  # pokračujte dalším sloupcem
            
            # Pro poplatky - použijeme specifické řazení
            if column in fee_columns:
                sorted_fee_values = sorted(df[column].dropna().unique(), key=convert_fee_to_float_simple)
                user_fee_input = right.multiselect(
                    column,
                    sorted_fee_values,
                    default=list(sorted_fee_values)
                )
                df = df[df[column].isin(user_fee_input)]
                continue  # pokračujte dalším sloupcem
            
            # Pro Min. investice
            if column == "Min. investice":
                unique_values = [val for val in df[column].dropna().unique() if val != "1 mil. Kč nebo 125 tis. euro"]
                user_cat_input = right.multiselect(
                    column,
                    unique_values,
                    default=list(unique_values)
                )
                if "1 mil. Kč" in user_cat_input:
                    user_cat_input.append("1 mil. Kč nebo 125 tis. euro")
                df = df[df[column].isin(user_cat_input)]
                continue  # pokračujte dalším sloupcem

            if column == "Lhůta pro zpětný odkup":
                    unique_values = sorted(df[column].dropna().unique(), key=extract_number_from_string)
                    user_cat_input = right.multiselect(
                    column,
                    unique_values,
                    default=list(unique_values)
                )
                    df = df[df[column].isin(user_cat_input)]

            if df[column].apply(lambda x: not pd.api.types.is_number(x)).any():
                unique_values = df[column].dropna().unique()

            elif is_numeric_dtype(df[column]):
                _min = df[column].min()
                _max = df[column].max()
                if pd.notna(_min) and pd.notna(_max) and _min != _max:
                    _min = float(_min)
                    _max = float(_max)
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




df.rename(columns={"Výnos 2022 (v %)":"Výnos 2022","Výnos 2021 (v %)":"Výnos 2021","Výnos 2020 (v %)":"Výnos 2020","Výnos od založení (% p.a.)":"Výnos od založení","TER (v %)":"TER","LTV (v %)":"LTV","YIELD (v %)":"YIELD"},inplace=True)

# Configure the image column
image_column = st.column_config.ImageColumn(label="Poskytovatel", width="medium")
vynos22_column = st.column_config.NumberColumn(label="Výnos 2022", format="%.2f %%")
vynos21_column = st.column_config.NumberColumn(label="Výnos 2021", format="%.2f %%")
vynos20_column = st.column_config.NumberColumn(label="Výnos 2020", format="%.2f %%")
vynos_all_column = st.column_config.NumberColumn(label="Výnos od založení", format="%.2f %% p.a.")
vynosTER_column = st.column_config.NumberColumn(label="TER", format="%.2f %%")
vynosLTV_column = st.column_config.NumberColumn(label="LTV", format="%.2f %%")
vynosYIELD_column = st.column_config.NumberColumn(label="YIELD", format="%.2f %%")

pocet_nemov_column = st.column_config.ProgressColumn(label="Počet nemovitostí",format="%f", min_value=0,
            max_value=50)

nazev_column = st.column_config.TextColumn(label="Název fondu", width="small")

df.set_index('Poskytovatel', inplace=True)

# Display the filtered data

filtered_df = filter_dataframe(df)
filtered_df.sort_values("Výnos 2022",ascending=False,inplace=True)



if not filtered_df.empty:
    st.dataframe(filtered_df.drop(columns=["Rozložení portfolia"]), hide_index=True, 
                 column_config={"Poskytovatel": image_column,
                                "Výnos 2022":vynos22_column,
                                "Výnos 2021":vynos21_column,
                                "Výnos 2020":vynos20_column,
                                "Výnos od založení":vynos_all_column,
                                "TER (v %)":vynosTER_column,
                                "LTV (v %)":vynosLTV_column,
                                "YIELD (v %)": vynosYIELD_column,
                                "Počet nemovitostí":pocet_nemov_column,
                                "Název fondu":nazev_column
                                }, height=428)
else:
    st.warning("Žádná data neodpovídají zvoleným filtrům.")