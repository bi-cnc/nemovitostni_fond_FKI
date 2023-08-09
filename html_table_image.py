

import streamlit as st
import pandas as pd
import base64
from PIL import Image


# Load the data
@st.cache_data
def load_data():
    df = pd.read_csv("FKI_fondy_streamlit.csv")
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("Filtry")

df.info()



from io import BytesIO
import base64

def get_image_in_html(img_path):
    with open(img_path, "rb") as img_file:
        my_string = base64.b64encode(img_file.read()).decode('utf-8')
    return f'<img src="data:image/png;base64,{my_string}" width="50" >'

df_html = df.copy()
df_html["Poskytovatel"] = df_html["logo_path"].apply(lambda x: get_image_in_html(x))
html_table = df_html.to_html(escape=False)

st.write(html_table, unsafe_allow_html=True)



import pandas as pd
import streamlit as st

data_df = pd.DataFrame(
    {
        "apps": [
            "https://storage.googleapis.com/s4a-prod-share-preview/default/st_app_screenshot_image/5435b8cb-6c6c-490b-9608-799b543655d3/Home_Page.png",
            "https://storage.googleapis.com/s4a-prod-share-preview/default/st_app_screenshot_image/ef9a7627-13f2-47e5-8f65-3f69bb38a5c2/Home_Page.png",
            "https://storage.googleapis.com/s4a-prod-share-preview/default/st_app_screenshot_image/31b99099-8eae-4ff8-aa89-042895ed3843/Home_Page.png",
            "https://storage.googleapis.com/s4a-prod-share-preview/default/st_app_screenshot_image/6a399b09-241e-4ae7-a31f-7640dc1d181e/Home_Page.png",
        ],
    }
)

st.data_editor(
    data_df,
    column_config={
        "apps": st.column_config.ImageColumn(
            "Preview Image", help="Streamlit app preview screenshots"
        )
    },
    hide_index=True,
)