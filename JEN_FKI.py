
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

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            fullscreen {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)


st.title("Fondy kvalifikovaných investorů")

# Load the data
@st.cache_data
def load_data():
    df = pd.read_csv("FKI_fondy_streamlit.csv")
    return df

df = load_data()

df.rename(columns={'Rozložení portfolia':"Portfolio"},inplace=True)

df["Název fondu"] = df["Název fondu"] + " 💬"

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
df["Cílený roční výnos"].fillna("- - -", inplace=True)
df["Vstupní poplatek"].fillna("- - -", inplace=True)
df["Manažerský poplatek"].fillna("- - -", inplace=True)
df["Výkonnostní poplatek"].fillna("- - -", inplace=True)
df["Výstupní poplatek"].fillna("- - -", inplace=True)
df["Lhůta pro zpětný odkup"].fillna("- - -", inplace=True)
df["Portfolio"].fillna("- - -", inplace=True)


def convert_yield_to_float(yield_value):
    if yield_value == "- - -":
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
        "kancelářské": ["kancelářské", "kancelář","kanceláře", "administrativní","office"],
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
        if not numbers:  # pokud nejsou žádná čísla, vrátíme -1 (nebo jinou náhradní hodnotu)
            return -1

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
    return -1  # Pokud nedostaneme žádnou platnou hodnotu, vrátíme -1 (nebo jinou náhradní hodnotu)

df['Uživatelský výběr'] = False

df_original = df.copy()

fee_columns = ["Vstupní poplatek", "Manažerský poplatek", "Výkonnostní poplatek", "Výstupní poplatek"]


def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify1 = st.toggle("Přidat filtrování", key="checkbox1")

    if not modify1:
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

        columns_to_exclude = ["Portfolio", "Výnos 2022", "Výnos 2021", "Výnos 2020", "Výnos od založení", "TER", "LTV", "YIELD", "WAULT", "NAV (v mld. Kč)","Název fondu"]
        available_columns = [col for col in df.columns if col not in columns_to_exclude]
        to_filter_columns = st.multiselect("Filtrovat přehled podle:", available_columns, placeholder="Vybrat finanční ukazatel")

        if len(to_filter_columns) > 1:
            st.warning("V tomto filtru můžete vybrat pouze 1 finanční ukazatel. Rozsáhlejší filtrování je dostupné ve fullscreenu (⛶) aplikace.")
            to_filter_columns = []  # Reset the selection

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
                df["Uživatelský výběr"] = True
                continue
            
            if column == "Cílený roční výnos":
                user_yield_input = right.multiselect(
                    "Cílený roční výnos",
                    sorted_yield_values,
                    default=sorted_yield_values  # ve výchozím stavu označit všechny hodnoty
                )
                df = df[df["Cílený roční výnos"].isin(user_yield_input)]
                df["Uživatelský výběr"] = True
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
                df["Uživatelský výběr"] = True
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
                df["Uživatelský výběr"] = True
                continue  # pokračujte dalším sloupcem

            if column == "Lhůta pro zpětný odkup":
                    unique_values = sorted(df[column].dropna().unique(), key=extract_number_from_string)
                    user_cat_input = right.multiselect(
                    column,
                    unique_values,
                    default=list(unique_values)
                )
                    df = df[df[column].isin(user_cat_input)]
                    df["Uživatelský výběr"] = True

            if df[column].apply(lambda x: not pd.api.types.is_number(x)).any():
                unique_values = df[column].dropna().unique()

            elif is_numeric_dtype(df[column]):
                _min = df[column].min()
                _max = df[column].max()
                if pd.notna(_min) and pd.notna(_max):
                    _min = float(_min)
                    _max = float(_max)

                    # Použití st.number_input pro zadání rozsahu
                    user_num_input = right.number_input(
                        f"{column} - Zadejte minimální hodnotu",
                        min_value=_min,
                        max_value=_max,
                        value=_min,  # Nastavíme minimální hodnotu jako výchozí
                        step=0.01,   # Přizpůsobte krok podle vašich potřeb
                    )

                    # Získání zadané minimální hodnoty
                    min_val = user_num_input

                    user_num_input = right.number_input(
                        f"{column} - Zadejte maximální hodnotu",
                        min_value=_min,  # Přizpůsobte minimální hodnotu podle zadaného min_val
                        max_value=_max,
                        value=_max,      # Nastavíme maximální hodnotu jako výchozí
                        step=0.01,       # Přizpůsobte krok podle vašich potřeb
                    )

                    # Získání zadané maximální hodnoty
                    max_val = user_num_input

                    df = df[df[column].between(min_val, max_val)]
                    df["Uživatelský výběr"] = True

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




df.rename(columns={"Výnos 2022 (v %)":"Výnos 2022 ","Výnos 2021 (v %)":"Výnos 2021 ","Výnos 2020 (v %)":"Výnos 2020 ","Výnos od založení (% p.a.)":"Výnos od založení ","TER (v %)":"TER ","LTV (v %)":"LTV ","YIELD (v %)":"YIELD ",
                   "WAULT":"WAULT ","NAV (v mld. Kč)":"NAV "},inplace=True)


df.info()

def get_emoji(value):
    if value >= 10:
        return "🔹"
    elif value >= 5:
        return "🔸"
    elif value < 5:
        return "💢"
    else:
        return "▫"

import numpy as np

# Vytvořte nový sloupec kombinující emoji a hodnotu 'Výnos 2022'
df['Výnos 2022'] = df['Výnos 2022 '].apply(lambda x: f"{get_emoji(x)} {x:.2f} %" if not np.isnan(x) else "▫️ - - -")
df['Výnos 2021'] = df['Výnos 2021 '].apply(lambda x: f"{get_emoji(x)} {x:.2f} %" if not np.isnan(x) else "▫️ - - -")
df['Výnos 2020'] = df['Výnos 2020 '].apply(lambda x: f"{get_emoji(x)} {x:.2f} %" if not np.isnan(x) else "▫️ - - -")
df['Výnos od založení'] = df['Výnos od založení '].apply(lambda x: f"{get_emoji(x)} {x:.2f} % p.a." if not np.isnan(x) else "▫️ - - -")

df["TER"] = df["TER "].apply(lambda x: "- - -" if pd.isna(x) else f"{x:.2f} %")
df["LTV"] = df["LTV "].apply(lambda x: "- - -" if pd.isna(x) else f"{x:.2f} %")
df["YIELD"] = df["YIELD "].apply(lambda x: "- - -" if pd.isna(x) else f"{x:.2f} %")

df["WAULT"] = df["WAULT "].apply(lambda x: "- - -" if pd.isna(x) else f"{x:.2f}")
df["NAV (v mld. Kč)"] = df["NAV "].apply(lambda x: "- - -" if pd.isna(x) else f"{x:.2f}")



# Configure the image column
image_column = st.column_config.ImageColumn(label="Poskytovatel", width="medium")
min_invest_column = st.column_config.TextColumn(help="📍**Minimální nutná částka pro vstup do fondu.** Klíčové zejména u FKI, kde je většinou 1 mil. Kč při splnění testu vhodnosti, ale někdy i 2 a více milionů.")
poplatky_column = st.column_config.TextColumn(help="📍**Často přehlížené, ale pro finální výnos zásadní jsou poplatky.** Je třeba znát podmínky pro výstupní poplatky v různých časových horizontech – zejména ty může investor ovlivnit.")


vynosNAV_column = st.column_config.TextColumn(label="NAV (v mld. Kč) 💬",help="📍**NAV (AUM): Hodnota majetku fondu ukazuje na robustnost a vloženou důvěru investorů.**")
vynosTER_column = st.column_config.TextColumn(label="TER 💬", help="📍**TER: Celkové roční náklady na správu fondu.** Čím nižší, tím lepší pro investory.")
vynosLTV_column = st.column_config.TextColumn(label="LTV 💬", help="📍**LTV: Loan to value – poměr cizího kapitálu k hodnotě nemovitosti.** Vyšší LTV pomáhá fondům dosahovat vyšších výnosů, ale zároveň je třeba říct, že větší úvěrové zatížení s sebou nese i větší riziko, kdyby se nějak dramaticky zvedly úroky z úvěru nebo propadly příjmy z pronájmu ")
vynosYIELD_column = st.column_config.TextColumn(label="YIELD 💬", help="📍**YIELD: Poměr čistého ročního nájmu a hodnoty nemovitostí.** Pokud poměříte čistý roční nájem celkovou hodnotou nemovitostí, zjistíte, jakou rentabilitu ty nemovitosti mají, aneb jaké hrubé výnosy dokáže fond generovat z nájmu. Na detailu každého fondu najdete tento údaj již vypočtený pod ukazatelem „Yield“. Zpravidla to bývá mezi 5-7 % p.a. ")
vynosWAULT_column = st.column_config.TextColumn(label="WAULT (v letech) 💬", help="📍**WAULT: Průměrná doba do konce nájemních smluv.** Jak dlouhé má v průměru nájemní smlouvy, respektive jaká je průměrná vážená doba do konce platnosti nájemních smluv. Obecně lze říct, že čím delší doba do konce platnosti nájemních smluv, tím lépe, protože o to jistější má fond příjmy. Zpravidla to bývá mezi 3-7 lety.")

nazev_column = st.column_config.TextColumn(label="Název fondu 💬", width="medium", help="📍**Po kliknutí na fond zjistíte další podrobnosti.**")

pocet_nemov_column = st.column_config.NumberColumn(label="Počet nemovitostí")

rozlozeni_column = st.column_config.TextColumn(label="Rozložení portfolia")

df.set_index('Poskytovatel', inplace=True)


filtered_df = filter_dataframe(df)
filtered_df.sort_values("Výnos 2022",ascending=False,inplace=True)

# Seznam sloupců, které chcete přesunout na začátek
cols_to_move = ["Název fondu",'Výnos 2022','Výnos 2021',"Výnos 2020","Výnos od založení","Cílený roční výnos","Min. investice","Vstupní poplatek","Manažerský poplatek","Výkonnostní poplatek","Výstupní poplatek","TER","Lhůta pro zpětný odkup",
                "LTV","WAULT","YIELD","NAV (v mld. Kč)","Počet nemovitostí","Portfolio"]

# Získání seznamu všech sloupců v DataFrame a odstranění sloupců, které chcete přesunout na začátek
remaining_cols = [col for col in df.columns if col not in cols_to_move]

# Kombinování obou seznamů k vytvoření nového pořadí sloupců
new_order = cols_to_move + remaining_cols

# Přeuspořádání sloupců v DataFrame
filtered_df = filtered_df[new_order]

filtered_df.info()

if not filtered_df.empty:
    st.dataframe(filtered_df.drop(columns=["Rozložení portfolia","Výnos 2022 ","Výnos 2021 ","Výnos 2020 ","Výnos od založení ","TER ","LTV ","YIELD ","WAULT ","NAV ","Uživatelský výběr"]), hide_index=True, 
                 column_config={"Poskytovatel": image_column,
                                "TER":vynosTER_column,
                                "LTV":vynosLTV_column,
                                "YIELD": vynosYIELD_column,
                                "Počet nemovitostí":pocet_nemov_column,
                                "Název fondu":nazev_column,
                                "Portfolio":rozlozeni_column,
                                "NAV (v mld. Kč)":vynosNAV_column,
                                "WAULT":vynosWAULT_column,
                                "Min. investice":min_invest_column,
                                "Vstupní poplatek":poplatky_column,
                                "Manažerský poplatek":poplatky_column,
                                "Výkonnostní poplatek":poplatky_column,
                                "Výstupní poplatek":poplatky_column,
                                }, height=428)
else:
    st.warning("Žádná data neodpovídají zvoleným filtrům.")

from streamlit.components.v1 import html

# Styling
st.markdown("""
<style>
.portal-navigator {
    padding-left: .5em;
    display: flex;
    justify-items: center;
    align-items: center;
    justify-content: center;
    background-color: white;
    color: #404040;
    height: 1.5em;
    border-radius: 6px;
    position: absolute;
    top: 3px;
    right: 3px;
    opacity: 1;
    z-index: 999;
    filter: drop-shadow(rgba(0, 0, 0, 0.3) 0 2px 10px);
}

.portal-navigator > a {
    margin-right: .5em;
    color: #069;
    text-decoration: underline;
    cursor: pointer; /* Přidání kurzoru jako ruky pro odkazy */
}
</style>
""", unsafe_allow_html=True)

# Script to add links
html("""
<script>
function add_navigator_to_portal(doc) {
    portal = doc.getElementById('portal');
    observer = new MutationObserver(function(mutations, observer) {
        let entry = portal.querySelector('.clip-region');
        if (entry) {
            let text = entry.textContent;
            let span = document.createElement('span');
            span.className = "portal-navigator";
            if (text.includes("WOOD & Company podfond Retail 💬")) {
                span.innerHTML = '<a href="https://wood.cz/produkty/fondy/retail-podfond/" target="_blank" >Zobrazit podrobnosti o fondu</a>';
            } else if (text.includes("Jet Industrial Lease 💬")) {
                span.innerHTML = '<a href="https://www.jetinvestment.cz/fondy-jet/jet-industrial-lease/" target="_blank" >Zobrazit podrobnosti o fondu</a>';
            } else if (text.includes("REALIA FUND SICAV, a.s. REALIA Podfond Retail Parks 💬")) {
                span.innerHTML = '<a href="https://www.avantfunds.cz/cs/fondy/realia-fund-sicav-a-s/realia-podfond-retail-parks/" target="_blank" >Zobrazit podrobnosti o fondu</a>';
            } else if (text.includes("WOOD & Company Office 💬")) {
                span.innerHTML = '<a href="https://wood.cz/produkty/fondy/office-podfond/" target="_blank" >Zobrazit podrobnosti o fondu</a>';
            } else if (text.includes("Silverline Real Estate 💬")) {
                span.innerHTML = '<a href="https://silverlinere.com/cs" target="_blank" >Zobrazit podrobnosti o fondu</a>';
            } else if (text.includes("Fond Českého bydlení  💬")) {
                span.innerHTML = '<a href="https://www.fondbydleni.cz/" target="_blank" >Zobrazit podrobnosti o fondu</a>';
            } else if (text.includes("Accolade Industrial Fund A2 Dis (CZK) 💬")) {
                span.innerHTML = '<a href="https://accolade.eu/domains/accolade.eu/cs/fond?gclid=Cj0KCQjwqP2pBhDMARIsAJQ0CzrdKx3tzR9Qf1ABf2hfJEG-JcTnwooKnt2HdcZf2JlJfluSd37ii28aAphTEALw_wcB" target="_blank" >Zobrazit podrobnosti o fondu</a>';
            } else if (text.includes("ZDR Investments Real Estate FKI 💬")) {
                span.innerHTML = '<a href="https://www.zdrinvestments.cz/" target="_blank" >Zobrazit podrobnosti o fondu</a>';
            } else if (text.includes("TRIKAYA nemovitostní fond SICAV, a.s. 💬")) {
                span.innerHTML = '<a href="https://fond.trikaya.cz/" target="_blank" >Zobrazit podrobnosti o fondu</a>';
            } else if (text.includes("Nova Real Estate 💬")) {
                span.innerHTML = '<a href="https://www.redsidefunds.com/cs/fondy/nova-real-estate" target="_blank" >Zobrazit podrobnosti o fondu</a>';
            } else if (text.includes("DOMOPLAN SICAV, a.s.  💬")) {
                span.innerHTML = '<a href="https://www.domoplan.eu/cs/investice/domoplan-sicav-a-s-6MDviG" target="_blank" >Zobrazit podrobnosti o fondu</a>';
            }
            // Přidání onclick atributu pro okamžité otevření odkazu při kliknutí
            span.querySelector('a').setAttribute('onclick', 'window.open(this.href); return false;');
            cont = entry.parentElement;
            cont.insertBefore(span, entry);
            console.log("inserted");

        }
    });
    observer.observe(portal, {childList: true});
};
add_navigator_to_portal(parent.window.document)
</script>
""")
