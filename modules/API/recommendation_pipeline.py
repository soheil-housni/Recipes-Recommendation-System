import torch
import mlflow
from ..inference import RecipesRecommender
from .config import initialisation

def recommendation(user_id,
                   ratings,
                   items,
                   techniques_users,
                   n_items,
                   n_ratings,
                   n_recommended_recipes):

    
    train_df,recipes_embeddings,recipes_df,scaler_user,device,best_recommendation_model,max_len_items,n_techniques_users=initialisation()

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