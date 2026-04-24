import torch
import pandas as pd
import ast
import numpy as np
from .inference_preprocessing import InferencePreprocessingUsers

class RecipesRecommender():
    def __init__(self,
                 model,
                 device,
                 recipes_embeddings,
                 recipes_df,
                 train_df,
                 scaler_user,
                 max_len_items,
                 n_techniques_users):
        
        self.device=device

        self.model=model
        self.model=self.model.to(self.device)

        self.recipes_embeddings=recipes_embeddings
        self.recipes_embeddings=self.recipes_embeddings.to(self.device)

        self.train_df=train_df
        self.recipes_df=recipes_df
        self.scaler_user=scaler_user
        self.max_len_items=max_len_items
        self.n_techniques_users=n_techniques_users

        self.user_preprocessor=InferencePreprocessingUsers(train_df=self.train_df,
                                                           max_len_items=self.max_len_items,
                                                           n_techniques_users=self.n_techniques_users,
                                                           scaler_user=self.scaler_user)
        



    def get_recommendations(self,
                            user_id,
                            ratings,
                            items,
                            techniques_users,
                            n_items,
                            n_ratings,
                            n_recommended_recipes:int=5):
        
        
        recipes_df = self.recipes_df.copy()

        items,techniques_users,n_items_scaled,ratings_scaled,n_ratings_scaled=self.user_preprocessor.preprocessing(user_id=user_id,
                                                                                                                   ratings=ratings,
                                                                                                                   items=items,
                                                                                                                   techniques_users=techniques_users,
                                                                                                                   n_items=n_items,
                                                                                                                   n_ratings=n_ratings)
        items=items.to(self.device)
        techniques_users=techniques_users.to(self.device)
        n_items_scaled=n_items_scaled.to(self.device)
        ratings_scaled=ratings_scaled.to(self.device)
        n_ratings_scaled=n_ratings_scaled.to(self.device)

        with torch.inference_mode():
            self.model.eval()
            user_embedding=self.model.forward_users(items=items,
                                                techniques_users=techniques_users,
                                                n_items_scaled=n_items_scaled,
                                                ratings_scaled=ratings_scaled,
                                                n_ratings_scaled=n_ratings_scaled)
        
            user_embedding=user_embedding.reshape(1,-1)
            cos_sim_scores=torch.nn.functional.cosine_similarity(user_embedding,self.recipes_embeddings,dim=1).to(torch.float32)
            cos_sim_scores=pd.Series(cos_sim_scores.cpu().numpy())
            recipes_df["cos_sim_scores"]=cos_sim_scores
            recommended_recipes=recipes_df.nlargest(n=n_recommended_recipes,columns=["cos_sim_scores"])
        return recommended_recipes

