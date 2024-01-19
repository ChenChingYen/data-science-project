import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Data Summary",
)

custom_css = """
<style>
    th {
        text-align: left;
    }
</style>
"""

st.title("✈︎ Data Summary")

# data description
# eCommerce behaviour data from multicategory stores
# This file contains seven months of behavior data (from October 2019 to April 2020) from a large multi-category online store, capturing events that represent many-to-many relations between products and users.\
# The data was collected by the [Open CDP]https://rees46.com/en/open-cdp project, utilizing an open-source customer data platform.
# https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store?select=2019-Oct.csv

st.divider()
st.subheader("Data Description ↴")
st.caption("### eCommerce behaviour data from multicategory stores")
st.caption("This file contains seven months of behavior data (from October 2019 to April 2020) from a large multi-category online store, capturing events that represent many-to-many relations between products and users.")
st.caption("The data was collected by the [Open CDP](https://rees46.com/en/open-cdp) project, utilizing an open-source customer data platform.")
# data overviews

# property | Description
# event_time | Time when event happened at (in UTC).
# event_type | Only one kind of event:
# view - a user viewed a product
# cart - a user added a product to shopping cart
# remove_from_cart - a user removed a product from shopping cart
# purchase - a user purchased a product
# product_id | ID of a product
# category_id | Product's category ID
# category_code | Product's category taxonomy (code name) if it was possible to make it. Usually present for meaningful categories and skipped for different kinds of accessories.
# brand	| Downcased string of brand name. Can be missed.
# price | Float price of a product.
# user_id | Permanent user ID.
# user_session | Temporary user's session ID. Same for each user's session. Is changed every time user come back to online store from a long pause.

# number of records (total file size, total columns, total rows)
# statistics, facts

data_description = {
    'Property':[
        "event_time",
        "event_type",
        "product_id",
        "category_id",
        "category_code",
        "brand",
        "price",
        "user_id",
        "user_session",
        ],
     'Description':[
        "Time when event happened at (in UTC).",
        "Only one kind of event:<br><strong>view</strong> - <em>a user viewed a product</em><br><strong>cart</strong> - <em>a user added a product to shopping cart</em><br><strong>remove_from_cart</strong> - <em>a user removed a product from shopping cart</em><br><strong>purchase</strong> - <em>a user purchased a product</em>",
        "ID of a product",
        "Product's category ID",
        "Product's category taxonomy (code name) if it was possible to make it. Usually present for meaningful categories and skipped for different kinds of accessories.",
        "Downcased string of brand name. Can be missed.",
        "Float price of a product.",
        "Permanent user ID.",
        "Temporary user's session ID. Same for each user's session. Is changed every time user come back to online store from a long pause.",
         ]
    }

st.divider()
st.subheader("Data Overviews ↴")

df = pd.DataFrame(data_description)
st.markdown(df.to_html(escape=False), unsafe_allow_html=True)
st.markdown(custom_css, unsafe_allow_html=True)

number_of_rows = 100
file_size = 2.9
st.caption(f"Total number of rows: {number_of_rows} rows")
st.caption(f"Total file size: {file_size} GB")