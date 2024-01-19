import pandas as pd
from collections import defaultdict
import time

def content_based_recommendation(selected_user_id, top_n):

    start = time.time()

    # receive user_id and return all past user behaviors

    print("---> loading user preferences data")

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

    min_value = min(list(user_preferences.values()))
    max_value = max(list(user_preferences.values()))
    for tag, score in user_preferences.items():
        user_preferences[tag] = (score - min_value) / (max_value - min_value)

    sorted(user_preferences.items(), key=lambda x:x[1], reverse=True)

    # get the popular products under each category (precalculated)

    print("---> calculating product scores")

    tag_hots = pd.read_csv(file_path_prefix + "popular_product_csv.csv")

    target_products = defaultdict(float)

    for user_category, user_score in user_preferences.items():
        for _, row in tag_hots[tag_hots['category'] == user_category].iterrows():
            target_products[row["product_id"]] += user_score * row["rating"]

    sorted(target_products.items(), key=lambda x:x[1], reverse=True)[:10]

    # filter out the products already purchased by the user

    print("---> filtering user purchased products")

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
    execution_time = end-start

    print("---> completed")

    print(f"execution time: {execution_time}s")
    # print(result_df)
    print(df_result)

    # df_result -> top 20 products recommended
    # execution_time -> total execution time

    return df_result, execution_time

# content_based_recommendation(567950899, 20)