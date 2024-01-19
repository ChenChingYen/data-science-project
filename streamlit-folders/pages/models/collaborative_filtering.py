import pyarrow.parquet as pq
import pandas as pd
from scipy.spatial import distance
import time

def collaborative_recommendation(selected_user_id, top_n):

    start = time.time()

    # select user embedding

    print("---> select user embedding")

    file_path_prefix = "E:/data_science_project/dataset/" # need to change to relative path before deployment

    user_parquet = pq.read_table(file_path_prefix + "spark-model/userFactors.parquet")
    item_parquet = pq.read_table(file_path_prefix + "spark-model/itemFactors.parquet")

    user_embedding_df = user_parquet.to_pandas()
    item_embedding_df = item_parquet.to_pandas()

    # user_embedding = list(user_embedding_df[user_embedding_df['id']==selected_user_id]['features'])[0]
    selected_user_embedding = user_embedding_df[user_embedding_df['id']==selected_user_id]

    # calculate cosine similarity between selected user id and all products

    print("---> calculate cosine similarity between user and products")

    item_embedding_df["similarity"] = item_embedding_df["features"].apply(
        lambda x: 1 - distance.cosine(selected_user_embedding.iloc[0]["features"], x)
    )

    # filter out purchased products and display the top N recommendations

    print("---> filtering user purchased products")

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

    products_df = pd.DataFrame(pd.read_csv(file_path_prefix + "sampled-product-dataset.csv"))

    df_result = pd.merge(
        left = df_target_product_id,
        right = products_df,
        left_on = "id",
        right_on = "product_id"
    )[["product_id", "category_code", "brand", "price", "similarity"]]

    # calculate execution time

    end = time.time()
    execution_time = end-start

    print("---> completed")

    print(f"execution time: {execution_time}s")

    # print(df_result)

    return df_result, execution_time
    # df_result -> top 20 products recommended
    # execution_time -> total execution time

collaborative_recommendation(567950899, 20)