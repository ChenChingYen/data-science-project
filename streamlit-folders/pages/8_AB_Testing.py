import streamlit as st
import pandas as pd
import time

# libraries and modules for content-based filtering
from collections import defaultdict

# libraries and modules for collaborative filtering
import pyarrow.parquet as pq
import pandas as pd
from scipy.spatial import distance

# libraries and modules for collaborative filtering
from tensorflow.keras.models import load_model

mock_result_a = pd.DataFrame({
    "product_id": [
        "a00", "a01", "a02", "a03", "a04", "a05", "a06", "a07", "a08", "a09",
        "a10", "a11", "a12", "a13", "a14", "a15", "a16", "a17", "a18", "a19",
        "a20", "a21", "a22", "a23", "a24", "a25", "a26", "a27", "a28", "a29",
        "a30", "a31", "a32", "a33", "a34", "a35", "a36", "a37", "a38", "a39",
        "a40", "a41", "a42", "a43", "a44", "a45", "a46", "a47", "a48", "a49",
        "a50", "a51", "a52", "a53", "a54", "a55", "a56", "a57", "a58", "a59",
        "a60", "a61", "a62", "a63", "a64", "a65", "a66", "a67", "a68", "a69",
        "a70", "a71", "a72", "a73", "a74", "a75", "a76", "a77", "a78", "a79",
        "a80", "a81", "a82", "a83", "a84", "a85", "a86", "a87", "a88", "a89",
        "a90", "a91", "a92", "a93", "a94", "a95", "a96", "a97", "a98", "a99",
    ],
    "category": [
        "ca00", "ca01", "ca02", "ca03", "ca04", "ca05", "ca06", "ca07", "ca08", "ca09",
        "ca10", "ca11", "ca12", "ca13", "ca14", "ca15", "ca16", "ca17", "ca18", "ca19",
        "ca20", "ca21", "ca22", "ca23", "ca24", "ca25", "ca26", "ca27", "ca28", "ca29",
        "ca30", "ca31", "ca32", "ca33", "ca34", "ca35", "ca36", "ca37", "ca38", "ca39",
        "ca40", "ca41", "ca42", "ca43", "ca44", "ca45", "ca46", "ca47", "ca48", "ca49",
        "ca50", "ca51", "ca52", "ca53", "ca54", "ca55", "ca56", "ca57", "ca58", "ca59",
        "ca60", "ca61", "ca62", "ca63", "ca64", "ca65", "ca66", "ca67", "ca68", "ca69",
        "ca70", "ca71", "ca72", "ca73", "ca74", "ca75", "ca76", "ca77", "ca78", "ca79",
        "ca80", "ca81", "ca82", "ca83", "ca84", "ca85", "ca86", "ca87", "ca88", "ca89",
        "ca90", "ca91", "ca92", "ca93", "ca94", "ca95", "ca96", "ca97", "ca98", "ca99",
    ],
    "price": [
        10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 
        10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 
        10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 
        10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 
        10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 
        10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 
        10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 
        10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 
        10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 
        10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 
    ]
})

mock_result_b = pd.DataFrame({
    "product_id": [
        "b00", "b01", "b02", "b03", "b04", "b05", "b06", "b07", "b08", "b09",
        "b10", "b11", "b12", "b13", "b14", "b15", "b16", "b17", "b18", "b19",
        "b20", "b21", "b22", "b23", "b24", "b25", "b26", "b27", "b28", "b29",
        "b30", "b31", "b32", "b33", "b34", "b35", "b36", "b37", "b38", "b39",
        "b40", "b41", "b42", "b43", "b44", "b45", "b46", "b47", "b48", "b49",
        "b50", "b51", "b52", "b53", "b54", "b55", "b56", "b57", "b58", "b59",
        "b60", "b61", "b62", "b63", "b64", "b65", "b66", "b67", "b68", "b69",
        "b70", "b71", "b72", "b73", "b74", "b75", "b76", "b77", "b78", "b79",
        "b80", "b81", "b82", "b83", "b84", "b85", "b86", "b87", "b88", "b89",
        "b90", "b91", "b92", "b93", "b94", "b95", "b96", "b97", "b98", "b99",
    ],
    "category": [
        "cba00", "cba01", "cba02", "cba03", "cba04", "cba05", "cba06", "cba07", "cba08", "cba09",
        "cba10", "cba11", "cba12", "cba13", "cba14", "cba15", "cba16", "cba17", "cba18", "cba19",
        "cba20", "cba21", "cba22", "cba23", "cba24", "cba25", "cba26", "cba27", "cba28", "cba29",
        "cba30", "cba31", "cba32", "cba33", "cba34", "cba35", "cba36", "cba37", "cba38", "cba39",
        "cba40", "cba41", "cba42", "cba43", "cba44", "cba45", "cba46", "cba47", "cba48", "cba49",
        "cba50", "cba51", "cba52", "cba53", "cba54", "cba55", "cba56", "cba57", "cba58", "cba59",
        "cba60", "cba61", "cba62", "cba63", "cba64", "cba65", "cba66", "cba67", "cba68", "cba69",
        "cba70", "cba71", "cba72", "cba73", "cba74", "cba75", "cba76", "cba77", "cba78", "cba79",
        "cba80", "cba81", "cba82", "cba83", "cba84", "cba85", "cba86", "cba87", "cba88", "cba89",
        "cba90", "cba91", "cba92", "cba93", "cba94", "cba95", "cba96", "cba97", "cba98", "cba99",
    ],
    "price": [
        20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
        20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 
        20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 
        20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 
        20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 
        20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 
        20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 
        20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 
        20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 
        20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 
    ]
})

@st.cache_data
def search_user_past_bahavior(selected_user_id):

    selected_user_id = int(selected_user_id)
    chunk_size = 1000000
    selected_user_df = pd.DataFrame(columns=['event_time', 'event_type', 'product_id', 'category_code', 'brand', 'price'])
    file_path_prefix = "E:/data_science_project/pages/models/dataset/"
    filename_list = [
        "filtered-sampled-cleaned-2019-Oct.csv",
        "filtered-sampled-cleaned-2019-Nov.csv",
        "filtered-sampled-cleaned-2019-Dec.csv",
        "filtered-sampled-cleaned-2020-Jan.csv",
        "filtered-sampled-cleaned-2020-Feb.csv",
        "filtered-sampled-cleaned-2020-Mar.csv",
        "filtered-sampled-cleaned-2020-Apr.csv"
        ]
    for filename in filename_list:
        path = file_path_prefix + filename
        chunks = pd.read_csv(path, chunksize=chunk_size, header=0)
        for chunk in chunks:
            selected_user_row = chunk.loc[chunk['user_id'] == selected_user_id, ['event_time', 'event_type', 'product_id', 'category_code', 'brand', 'price']]
            selected_user_df = pd.concat([selected_user_df, selected_user_row], ignore_index=True)

    print(f"---> {len(selected_user_df)} rows found")

    return selected_user_df

@st.cache_data
def search_user_preferences(selected_user_id):

    selected_user_id = int(selected_user_id)

    file_path_prefix = "E:/data_science_project/pages/models/dataset/"
    filename = "user_preffered_category.csv"

    preferences_df = pd.read_csv(file_path_prefix + filename)
    selected_user_row = preferences_df.loc[preferences_df['user_id'] == selected_user_id, ['category', 'rating']]

    print(f"---> {len(selected_user_row)} rows found")

    return selected_user_row

@st.cache
def model_a_prediction(selected_user_id, top_n, model_name):
    
    start = time.time()

    # receive user_id and return all past user behaviors

    selected_user_id = int(selected_user_id)

    print(f"---> loading user preferences data ({selected_user_id})")
    progress_placeholder.markdown(f"({model_name}) loading user preferences data ...")


    file_path_prefix = "E:/data_science_project/pages/models/dataset/" # need to change to relative path before deployment

    user_preferences_df = pd.read_csv(file_path_prefix + "user_preffered_category.csv")

    selected_user_preferences = user_preferences_df[user_preferences_df['user_id'] == selected_user_id]

    user_preferences = defaultdict(int)

    for _, row in selected_user_preferences.iterrows():
        user_preferences[row['category']] = row["rating"]
        
    del user_preferences_df
    del selected_user_preferences

    # Normalise the rating

    print("---> normalising user rating")
    progress_placeholder.markdown(f"({model_name}) normalising user rating ...")

    min_value = min(list(user_preferences.values()))
    max_value = max(list(user_preferences.values()))
    for tag, score in user_preferences.items():
        user_preferences[tag] = (score - min_value) / (max_value - min_value)

    sorted(user_preferences.items(), key=lambda x:x[1], reverse=True)

    # get the popular products under each category (precalculated)

    print("---> calculating product scores")
    progress_placeholder.markdown(f"({model_name}) calculating product scores ...")

    tag_hots = pd.read_csv(file_path_prefix + "popular_product_csv.csv")

    target_products = defaultdict(float)

    for user_category, user_score in user_preferences.items():
        for _, row in tag_hots[tag_hots['category'] == user_category].iterrows():
            target_products[row["product_id"]] += user_score * row["rating"]

    # sorted(target_products.items(), key=lambda x:x[1], reverse=True)[:10]

    # filter out the products already purchased by the user

    print("---> filtering user purchased products")
    progress_placeholder.markdown(f"({model_name}) filtering user purchased products ...")

    purchase_df = pd.DataFrame(pd.read_csv(file_path_prefix + "purchase_history.csv"))
    purchased_by_user_list = purchase_df[purchase_df["user_id"] == selected_user_id]["product_id"].tolist()

    df_result = pd.merge(
        left = pd.DataFrame(target_products.items(),
                            columns = ["product_id", "score"]),
        right = pd.read_csv(file_path_prefix + "sampled-product-dataset.csv"),
        on = "product_id"
    )

    df_result = df_result[~df_result['product_id'].isin(purchased_by_user_list)]

    df_result = df_result.sort_values(by="score", ascending=False).head(top_n)[["product_id", "category_code", "brand", "price", "score"]]

    # calculate execution time

    end = time.time()
    execution_time = round(end-start, 2)

    print("---> completed")
    progress_placeholder.markdown(f"({model_name}) completed ...")

    print(f"execution time: {execution_time}s")

    # df_result -> top 20 products recommended
    # execution_time -> total execution time

    return df_result.drop('score', axis=1), execution_time

@st.cache
def model_b_prediction(selected_user_id, top_n, model_name):
    
    start = time.time()

    selected_user_id = int(selected_user_id)

    # select user embedding

    print("---> select user embedding")
    progress_placeholder.markdown(f"({model_name}) select user embedding ...")

    file_path_prefix = "E:/data_science_project/pages/models/dataset/" # need to change to relative path before deployment

    user_parquet = pq.read_table(file_path_prefix + "spark-model/userFactors.parquet")
    item_parquet = pq.read_table(file_path_prefix + "spark-model/itemFactors.parquet")

    user_embedding_df = user_parquet.to_pandas()
    item_embedding_df = item_parquet.to_pandas()

    # user_embedding = list(user_embedding_df[user_embedding_df['id']==selected_user_id]['features'])[0]
    selected_user_embedding = user_embedding_df[user_embedding_df['id']==selected_user_id]

    # calculate cosine similarity between selected user id and all products

    print("---> calculate cosine similarity between user and products")
    progress_placeholder.markdown(f"({model_name}) calculate cosine similarity between user and products ...")

    item_embedding_df["similarity"] = item_embedding_df["features"].apply(
        lambda x: 1 - distance.cosine(selected_user_embedding.iloc[0]["features"], x)
    )

    # filter out purchased products and display the top N recommendations

    print("---> filtering user purchased products")
    progress_placeholder.markdown(f"({model_name}) filtering user purchased products ...")

    purchase_df = pd.DataFrame(pd.read_csv(file_path_prefix + "purchase_history.csv"))
    purchased_by_user_list = purchase_df[purchase_df["user_id"] == selected_user_id]["product_id"].tolist()

    df_target_product_id = (
        item_embedding_df[~item_embedding_df["id"]
                            .isin(purchased_by_user_list)]
        .sort_values(by="similarity", ascending=False)
        .head(top_n)[["id", "similarity"]]
    )

    # display product details

    print("---> display recommended product details")
    progress_placeholder.markdown(f"({model_name}) display recommended product details ...")

    products_df = pd.DataFrame(pd.read_csv(file_path_prefix + "sampled-product-dataset.csv"))

    df_result = pd.merge(
        left = df_target_product_id,
        right = products_df,
        left_on = "id",
        right_on = "product_id"
    )[["product_id", "category_code", "brand", "price", "similarity"]]

    # calculate execution time

    end = time.time()
    execution_time = round(end-start, 2)

    print("---> completed")

    print(f"execution time: {execution_time}s")

    # print(df_result)

    return df_result.drop('similarity', axis=1), execution_time
    # df_result -> top 20 products recommended
    # execution_time -> total execution time

st.set_page_config(
    page_title="A/B Testing",
    page_icon="✅",
    layout="wide"
)

results = {"a": [], "b": []}

st.title("☑ A/B Testing")
st.write("Select the products you most likely to choose")

st.markdown("""
<style>
.stAlert{
    display: none !important;
}
</style>""", unsafe_allow_html=True)

selected_user_id = 567950899

st.write(f"Selected User ID: {selected_user_id}")

st.markdown("---")
with st.spinner("loading user behaviors data ..."): user_behavior_data = search_user_past_bahavior(selected_user_id)
if len(user_behavior_data) <= 0: st.warning(f'user id {selected_user_id} not found', icon="⚠️")


user_data_table, user_preferences_table = st.columns([0.6, 0.4], gap="small")

with user_data_table:
    # user_behavior_data = search_user_past_bahavior(selected_user_id)
    user_data_table.subheader("User Behavior Data")
    user_data_table.dataframe(user_behavior_data)

with user_preferences_table:
    with st.spinner("loading user preferences data ..."):user_preferences = search_user_preferences(selected_user_id)
    user_preferences_table.subheader("User Preferences")
    user_preferences_table.dataframe(user_preferences)

st.subheader("A/B Testing")
testing_placeholder = st.empty()
progress_placeholder = st.empty()

# st.subheader("A/B Testing")
# progress_placeholder = st.empty()
# testing_placeholder = st.empty()

selected_product_df = pd.DataFrame(columns=["product_id", "category_code", "brand", "price", "click"])

def update_model_scores(selected_product, subset_a, subset_b):

    selected_product_id = selected_product["product_id"]

    if len(results["a"]) == 0 and len(results["b"]) == 0: score_a, score_b = 0, 0
    else: score_a, score_b = results["a"][-1], results["b"][-1]
    
    if selected_product_id.isin(subset_a["product_id"]).any():results["a"].append(score_a+1)
    else: results["a"].append(score_a)

    if selected_product_id.isin(subset_b["product_id"]).any(): results["b"].append(score_b+1)
    else: results["b"].append(score_b)

    print(f"a: {results['a'][-1]}", end=" ")
    print(f"b: {results['b'][-1]}")

# result_model_a = mock_result_a
# result_model_b = mock_result_b

result_model_a, _ = model_a_prediction(selected_user_id, 100, "content-based filtering model")
result_model_b, _ = model_b_prediction(selected_user_id, 100, "collaborative filtering model")

for i in range(0, 101, 5):
    progress_placeholder.caption(f"### test ({int(i/5)}/20)")
    print(f"test {int(i/5)}")
    # print("---> dataframe: ")
    temp = pd.concat([result_model_a.iloc[i:i+5], result_model_b.iloc[i:i+5]])
    temp["click"] = False
    # print(mock_result_a.iloc[i:i+10])
    # print(mock_result_b.iloc[i:i+10])
    # print(temp)
    
    edited_df = testing_placeholder.data_editor(
        temp,
        disabled=["widgets"],
        hide_index=True,
        key=f"testing_table_{i}"
    )
    # selectbox_table.button("Submit", type="primary", on_click=click_handler)
    # st.button("Submit", type="primary", key="submit_button", on_click=restart)
    
    selected_product = edited_df[edited_df["click"]==True]
    # print(selected_product)
    
    while i<100:
        if len(selected_product) < 1:
            selected_product = edited_df[edited_df["click"]==True]
            time.sleep(1)
            continue
        selected_product_df = pd.concat([selected_product_df, selected_product])
        update_model_scores(selected_product,
                            result_model_a.iloc[i:i+5],
                            result_model_b.iloc[i:i+5])
        # print(f"{len(selected_product_df)} products selected")
        break
        # print(f"{len(edited_df[edited_df['click']==True])} product selected")

testing_placeholder.dataframe(selected_product_df)

st.divider()

if len(selected_product_df) >= 10:

    statement = ""

    if results['a'][-1] > results['b'][-1]: statement = "**Model A** has a higher click-through rate (CTR)"
    elif results['b'][-1] > results['a'][-1]: statement = "**Model B** has a higher click-through rate (CTR)"
    else: statement = "**Both Models** have the same click-through rate (CTR)"

    st.markdown(f"Score for Model A: **{results['a'][-1]}** / Score for Model B: **{results['b'][-1]}**. {statement}")

    chart_data = pd.DataFrame(results)
    st.line_chart(chart_data)
