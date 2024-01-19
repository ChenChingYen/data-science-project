import streamlit as st
from PIL import Image
import pandas as pd

st.set_page_config(
    page_title="User Behaviour Analysis",
    page_icon="📈",
    layout="wide"
)

popularity_image_mapping = {
    'October 2019': "top_bottom_2019_Oct_01.png",
    'November 2019': "top_bottom_2019_Nov_01.png",
    'December 2019': "top_bottom_2019_Dec_01.png",
    'January 2020': "top_bottom_2020_Jan_01.png",
    'February 2020': "top_bottom_2020_Feb_01.png",
    'March 2020': "top_bottom_2020_Mar_01.png",
    'April 2020': "top_bottom_2020_Apr_01.png",
}

time_series_image_mapping = {
    'October 2019': "_Oct_2019.png",
    'November 2019': "_Nov_2019.png",
    'December 2019': "_Dec_2019.png",
    'January 2020': "_Jan_2020.png",
    'February 2020': "_Feb_2020.png",
    'March 2020': "_Mar_2020.png",
    'April 2020': "_Apr_2020.png",
}

def render_image(month):
    chart = Image.open(chart_image_path_prefix + popularity_image_mapping[month])
    chart_holder.image(chart, use_column_width="always", output_format='png')

def render_time_series_image(month):
    chart_01 = Image.open(chart_image_path_prefix + "view" + time_series_image_mapping[month])
    chart_02 = Image.open(chart_image_path_prefix + "purchase" + time_series_image_mapping[month])
    time_series_chart_holder_01.image(chart_01, use_column_width="always", output_format='png')
    time_series_chart_holder_02.image(chart_02, use_column_width="always", output_format='png')

st.title("✈︎ User Behaviour Analysis")

#####

st.divider()
st.subheader("Product Popularity")

chart_image_path_prefix = "E:/data_science_project/pages/visualizations/dark/"

popular_month = st.selectbox(
    'Select a Month',
    options=['October 2019', 'November 2019', 'December 2019', 'January 2020', 'February 2020', 'March 2020', 'April 2020'],
    key="popularity"
    # on_change=render,
    # args="selected_month"
    )

chart_holder = st.empty()
render_image(popular_month)

#####

st.divider()
st.subheader("User Behaviors")

behavior_data = {
    "October 2019": pd.DataFrame({
        "Event Type": ['view', 'cart', 'purchase'],
        "Frequency": [2520466, 80735, 54861],
        "Percentage": [94.894848, 3.039650, 2.065501]
    }), 
    "November 2019": pd.DataFrame({
        "Event Type": ['view', 'cart', 'purchase'],
        "Frequency": [3931702, 211407, 54861],
        "Percentage": [93.421684, 5.023269, 1.555047]
    }), 
    "December 2019": pd.DataFrame({
        "Event Type": ['view', 'cart', 'purchase'],
        "Frequency": [1504233, 56769, 18087],
        "Percentage": [95.259545, 3.595048, 1.145407]
    }), 
    "January 2020": pd.DataFrame({
        "Event Type": ['view', 'cart', 'purchase'],
        "Frequency": [1313126, 45469, 12934],
        "Percentage": [95.741760, 3.315205, 0.943035]
    }), 
    "February 2020": pd.DataFrame({
        "Event Type": ['view', 'cart', 'purchase'],
        "Frequency": [1327388, 54599, 21618],
        "Percentage": [94.569911, 3.889912, 1.540177]
    }), 
    "March 2020": pd.DataFrame({
        "Event Type": ['view', 'cart', 'purchase'],
        "Frequency": [1459310, 65733, 22577],
        "Percentage": [94.293819,  4.247360, 1.458821]
    }), 
    "April 2020": pd.DataFrame({
        "Event Type": ['view', 'cart', 'purchase'],
        "Frequency": [1888532, 81669, 24224],
        "Percentage": [94.690550,  4.094864, 1.214586]
    }), 
}

stat_left, stat_right = st.columns(2, gap="large")

stat_holder = stat_left.empty()
stat_holder.table(behavior_data[popular_month])

event_type_image_path = "E:/data_science_project/pages/visualizations/dark/event_type_by_month.png"
stat_right.image(Image.open(event_type_image_path), use_column_width="always", output_format='png')

#####

st.divider()
st.subheader("Time Series Analysis")

time_series_left, time_series_right = st.columns(2, gap="small")

time_series_chart_holder_01 = time_series_left.empty()
time_series_chart_holder_02 = time_series_right.empty()
render_time_series_image(popular_month)