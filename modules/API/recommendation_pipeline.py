from ..inference_recipes_recommender import RecipesRecommender
from ..load_parquet_file import load
import joblib
import pyarrow.parquet as pq
import torch
import mlflow


def recommendation(user_id,
                   ratings,
                   items,
                   techniques_users,
                   n_items,
                   n_ratings,
                   n_recommended_recipes):

    train_parquet_file=pq.ParquetFile("../dataframes/train_df.parquet")
    train_df=load(train_parquet_file)

    recipes_embeddings=torch.load("../recipes_set/recipes_embeddings.pt")

    recipes_parquet_file=pq.ParquetFile("../recipes_set/recipes_df.parquet")
    recipes_df=load(recipes_parquet_file)

    scaler_user=joblib.load("../scalers/scaler_users.pkl")

    device = (torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    best_model_registered_uri=f"models:/Best_Recommendation_System_model/1"
    best_recommendation_model=mlflow.pytorch.load_model(best_model_registered_uri,map_location=torch.device("cpu"))
    best_recommendation_model.device=device
    max_len_items=6437
    n_techniques_users=58

    recipes_recommender=RecipesRecommender(device=device,model=best_recommendation_model,recipes_embeddings=recipes_embeddings,recipes_df=recipes_df,train_df=train_df,scaler_user=scaler_user,max_len_items=max_len_items,n_techniques_users=n_techniques_users)

    recommended_recipes=recipes_recommender.get_recommendations(
        user_id=user_id,
        ratings=ratings,
        items=items,
        techniques_users=techniques_users,
        n_items=n_items,
        n_ratings=n_ratings,
        n_recommended_recipes=n_recommended_recipes
    )

    return recommended_recipes