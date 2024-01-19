import streamlit as st
import pandas as pd
import time
# from models.content_based_filtering import content_based_recommendation

# libraries and modules for content-based filtering
from collections import defaultdict

# libraries and modules for collaborative filtering
import pyarrow.parquet as pq
import pandas as pd
from scipy.spatial import distance

# libraries and modules for collaborative filtering
from tensorflow.keras.models import load_model

# @st.cache_data
def search_user_past_bahavior(selected_user_id):

    start = time.time()

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
        progress_update_behaviors.markdown(f"loading {filename} ...")
        path = file_path_prefix + filename
        chunks = pd.read_csv(path, chunksize=chunk_size, header=0)
        for chunk in chunks:
            selected_user_row = chunk.loc[chunk['user_id'] == selected_user_id, ['event_time', 'event_type', 'product_id', 'category_code', 'brand', 'price']]
            selected_user_df = pd.concat([selected_user_df, selected_user_row], ignore_index=True)

    print(f"---> {len(selected_user_df)} rows found")

    # print(selected_user_df)

    end = time.time()
    execution_time = round(end-start, 2)

    return selected_user_df, execution_time

# @st.cache_data
def search_user_preferences(selected_user_id):

    start = time.time()

    selected_user_id = int(selected_user_id)

    file_path_prefix = "E:/data_science_project/pages/models/dataset/"
    filename = "user_preffered_category.csv"
    progress_update_preferences.markdown(f"loading {filename} ...")

    preferences_df = pd.read_csv(file_path_prefix + filename)
    selected_user_row = preferences_df.loc[preferences_df['user_id'] == selected_user_id, ['category', 'rating']]

    print(f"---> {len(selected_user_row)} rows found")

    # print(selected_user_df)

    end = time.time()
    execution_time = round(end-start, 2)

    return selected_user_row, execution_time

def content_based_recommendation(selected_user_id, top_n):

    start = time.time()

    # receive user_id and return all past user behaviors

    selected_user_id = int(selected_user_id)

    print(f"---> loading user preferences data ({selected_user_id})")
    progress_update_1.markdown("loading user preferences data ...")


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
    progress_update_1.markdown("normalising user rating ...")

    min_value = min(list(user_preferences.values()))
    max_value = max(list(user_preferences.values()))
    for tag, score in user_preferences.items():
        user_preferences[tag] = (score - min_value) / (max_value - min_value)

    sorted(user_preferences.items(), key=lambda x:x[1], reverse=True)

    # get the popular products under each category (precalculated)

    print("---> calculating product scores")
    progress_update_1.markdown("calculating product scores ...")

    tag_hots = pd.read_csv(file_path_prefix + "popular_product_csv.csv")

    target_products = defaultdict(float)

    for user_category, user_score in user_preferences.items():
        for _, row in tag_hots[tag_hots['category'] == user_category].iterrows():
            target_products[row["product_id"]] += user_score * row["rating"]

    # sorted(target_products.items(), key=lambda x:x[1], reverse=True)[:10]

    # filter out the products already purchased by the user

    print("---> filtering user purchased products")
    progress_update_1.markdown("filtering user purchased products ...")

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
    progress_update_1.markdown("completed ...")

    print(f"execution time: {execution_time}s")

    # df_result -> top 20 products recommended
    # execution_time -> total execution time

    return df_result, execution_time

def collaborative_recommendation(selected_user_id, top_n):

    start = time.time()

    selected_user_id = int(selected_user_id)

    # select user embedding

    print("---> select user embedding")
    progress_update_2.markdown("select user embedding ...")

    file_path_prefix = "E:/data_science_project/pages/models/dataset/" # need to change to relative path before deployment

    user_parquet = pq.read_table(file_path_prefix + "spark-model/userFactors.parquet")
    item_parquet = pq.read_table(file_path_prefix + "spark-model/itemFactors.parquet")

    user_embedding_df = user_parquet.to_pandas()
    item_embedding_df = item_parquet.to_pandas()

    # user_embedding = list(user_embedding_df[user_embedding_df['id']==selected_user_id]['features'])[0]
    selected_user_embedding = user_embedding_df[user_embedding_df['id']==selected_user_id]

    # calculate cosine similarity between selected user id and all products

    print("---> calculate cosine similarity between user and products")
    progress_update_2.markdown("calculate cosine similarity between user and products ...")

    item_embedding_df["similarity"] = item_embedding_df["features"].apply(
        lambda x: 1 - distance.cosine(selected_user_embedding.iloc[0]["features"], x)
    )

    # filter out purchased products and display the top N recommendations

    print("---> filtering user purchased products")
    progress_update_2.markdown("filtering user purchased products ...")

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
    progress_update_2.markdown("display recommended product details ...")

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

    return df_result, execution_time
    # df_result -> top 20 products recommended
    # execution_time -> total execution time

def dnn_recommendation(selected_user_id, top_n):

    start = time.time()

    selected_user_id = int(selected_user_id)

    # loading tensorflow model
    print("---> loading tensorflow model")
    progress_update_3.markdown("loading tensorflow model ...")

    file_path_prefix = "E:/data_science_project/pages/models/dataset/"

    recommendation_model = load_model(file_path_prefix + "tensorflow-model/model")
    user_embedding = pd.read_csv(file_path_prefix + "tensorflow-model/tensorflow-user-embedding.csv", header=None)
    product_embedding = pd.read_csv(file_path_prefix + "tensorflow-model/tensorflow-product-embedding.csv", header=None)

    # select user embedding data
    print("---> select user embedding data")
    progress_update_3.markdown("select user embedding data ...")

    selected_user_embedding = user_embedding[user_embedding[0] == selected_user_id]
    del user_embedding

    DNN_ITERATION_LIMIT = 200
    RANDOM_STATE = 42

    # select random product embedding group
    print("---> select random product embedding group")
    progress_update_3.markdown("select random product embedding group ...")

    result_df = pd.DataFrame(columns=["product_id", "predicted_rating"])
    sampled_product_embedding = product_embedding.sample(n=DNN_ITERATION_LIMIT, random_state=RANDOM_STATE)

    # predict user-product rating
    print("---> predict user-product rating")
    progress_update_3.markdown("predict user-product rating ...")

    for _, row in sampled_product_embedding.iterrows():

        row_product_embedding = row.to_frame().T

        prediction = recommendation_model.predict([
            (selected_user_embedding[1]), (selected_user_embedding[2]), (selected_user_embedding[3]), (selected_user_embedding[4]), (selected_user_embedding[5]),
            (selected_user_embedding[6]), (selected_user_embedding[7]), (selected_user_embedding[8]), (selected_user_embedding[9]), (selected_user_embedding[10]),
            (row_product_embedding[1]), (row_product_embedding[2]), (row_product_embedding[3]), (row_product_embedding[4]), (row_product_embedding[5]),
            (row_product_embedding[6]), (row_product_embedding[7]), (row_product_embedding[8]), (row_product_embedding[9]), (row_product_embedding[10]),
            (row_product_embedding[11]), (row_product_embedding[12]), (row_product_embedding[13]), (row_product_embedding[14]), (row_product_embedding[15]),
            (row_product_embedding[16]), (row_product_embedding[17]), (row_product_embedding[18]), (row_product_embedding[19])
        ])

        result_df = pd.concat([result_df, pd.DataFrame.from_records([{ 'product_id': str(row_product_embedding[0].iloc[0]).split('.')[0], 'predicted_rating': prediction[0][0] }])], ignore_index=True)

    # filter user purchased products
    print("---> filter user purchased products")
    progress_update_3.markdown("filter user purchased products ...")

    purchase_df = pd.DataFrame(pd.read_csv(file_path_prefix + "purchase_history.csv"))
    purchased_by_user_list = purchase_df[purchase_df["user_id"] == selected_user_id]["product_id"].tolist()

    df_target_product_id = (
        result_df[~result_df["product_id"].isin(purchased_by_user_list)]
            .sort_values(by="predicted_rating", ascending=False)
            .head(top_n)[["product_id", "predicted_rating"]]
    )

    # fetching recommended product details
    print("---> fetching recommended product details")
    progress_update_3.markdown("fetching recommended product details ...")

    products_df = pd.DataFrame(pd.read_csv(file_path_prefix + "sampled-product-dataset.csv"))

    df_target_product_id["product_id"] = df_target_product_id["product_id"].astype(str)
    products_df["product_id"] = products_df["product_id"].astype(str)

    df_result = pd.merge(
        left = df_target_product_id,
        right = products_df,
        on = "product_id"
        )[["product_id", "category_code", "brand", "price", "predicted_rating"]]

    # calculate execution time

    end = time.time()
    execution_time = round(end-start, 2)

    print("---> completed")

    print(f"execution time: {execution_time}s")

    return df_result, execution_time

st.set_page_config(
    page_title="Recommendation Model",
    page_icon="📲",
    layout="wide"
)

st.title("✈︎ Recommendation Model")

st.markdown("""---""")

input_left, input_center, input_right = st.columns(3, gap="large")

with input_left: selected_user_id = st.text_input("User ID: ", value="", placeholder="Enter a User ID")

with input_center: top_n = st.slider('Number of Recommendations: ', 10, 100, 10, step=10)

with input_right: selected_model = st.selectbox("Select Model: ", ["All", "Content-based Filtering", "Collaborative Filtering", "Deep Neural Network"], index=0)

# 567950899

if st.button("Recommend"):

    st.markdown("---")

    # display user past behaviors

    st.markdown(f"## Selected User Data ({selected_user_id})")

    tab1, tab2 = st.tabs(["Past Behaviors", "Preferences"])

    with tab1:
        st.header("Past User Behaviors")
        progress_update_behaviors = st.empty()
        progress_update_behaviors_table = st.empty()
        with st.spinner("loading user data ..."): selected_user_behaviors, execution_time = search_user_past_bahavior(selected_user_id)
        progress_update_behaviors.markdown(f"execution time: {execution_time}s")
        if len(selected_user_behaviors) > 0: progress_update_behaviors_table.dataframe(selected_user_behaviors)
        else: progress_update_behaviors_table.warning(f'user id {selected_user_id} not found', icon="⚠️")

    with tab2:
        st.header("User Preferences")
        progress_update_preferences = st.empty()
        progress_update_preferences_table = st.empty()
        with st.spinner("loading user data ..."): selected_user_preferences, execution_time = search_user_preferences(selected_user_id)
        progress_update_preferences.markdown(f"execution time: {execution_time}s")
        if len(selected_user_preferences) > 0:
            # plot a bar chart to visualise user preferences
            preferences_table, preferences_chart = progress_update_preferences_table.columns([0.2, 0.8], gap="large")
            with preferences_table: st.dataframe(selected_user_preferences)
            sorted_selected_user_preferences = selected_user_preferences.sort_values(by='rating', ascending=False)
            result_dict = sorted_selected_user_preferences.set_index('category')['rating'].to_dict()
            with preferences_chart: st.bar_chart(result_dict, height=400)
        else: progress_update_preferences_table.warning(f'user id {selected_user_id} not found', icon="⚠️")
    
    # progress_update_0 = st.empty()
    # with st.spinner("loading user data ..."): selected_user_df, execution_time = search_user_past_bahavior(selected_user_id)
    # progress_update_0.markdown(f"execution time: {execution_time}s")
    
    if len(selected_user_preferences) > 0:
        # progress_update_behaviors_table.table(selected_user_behaviors)
        # progress_update_preferences_table.table(selected_user_preferences)
        # st.dataframe(selected_user_behaviors)

        if selected_model == "All" or selected_model == "Content-based Filtering":

            st.markdown("---")
            st.markdown("## Output (Content-based Filtering)")

            progress_update_1 = st.empty()
            progress_update_1.markdown("loading function ...")
            
            with st.spinner(""): content_based_filtering_result, et1 = content_based_recommendation(selected_user_id, top_n)

            st.dataframe(content_based_filtering_result)

            progress_update_1.markdown(f"execution time: {et1}s")

        if selected_model == "All" or selected_model == "Collaborative Filtering":

            st.markdown("---")
            st.markdown("## Output (Collaborative Filtering)")

            progress_update_2 = st.empty()
            progress_update_2.markdown("loading function ...")
            
            with st.spinner(""): collaborative_filtering_result, et2 = collaborative_recommendation(selected_user_id, top_n)

            st.dataframe(collaborative_filtering_result)

            progress_update_2.markdown(f"execution time: {et2}s")
            
        if selected_model == "All" or selected_model == "Deep Neural Network":

            st.markdown("---")
            st.markdown("## Output (Deep Neural Network)")

            progress_update_3 = st.empty()
            progress_update_3.markdown("loading function ...")
            
            with st.spinner(""): dnn_recommendation_result, et3 = dnn_recommendation(selected_user_id, top_n)

            st.dataframe(dnn_recommendation_result)

            progress_update_3.markdown(f"execution time: {et3}s")

        