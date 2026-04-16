import torch
import pandas as pd
import ast
import numpy as np

class RecipesRecommender():
    def __init__(self,
                 model,
                 device,
                 recipes_embeddings,
                 recipes_df,
                 scaler_user,
                 max_len_items,
                 n_techniques_users):
        
        self.model=model
        self.device=device
        self.model=self.model.to(self.device)
        self.recipes_embeddings=recipes_embeddings
        self.recipes_embeddings=self.recipes_embeddings.to(self.device)
        self.recipes_df=recipes_df
        self.scaler_user=scaler_user
        self.max_len_items=max_len_items
        self.n_techniques_users=n_techniques_users

    def get_recommendations(self,
                            items,
                            techniques_users,
                            n_items,
                            ratings,
                            n_ratings,
                            n_recommended_recipes:int=5):
        
        
        items,techniques_users,n_items_scaled,n_ratings_scaled,ratings_scaled=self.preprocessing(items,
                                                                                                 techniques_users,
                                                                                                 n_items,
                                                                                                 ratings,
                                                                                                 n_ratings,
                                                                                                 n_recommended_recipes)
        
        recipes_df = self.recipes_df.copy()

        items=items.to(self.device)
        techniques_users=techniques_users.to(self.device)
        n_items_scaled=n_items_scaled.to(self.device)
        ratings_scaled=ratings_scaled.to(self.device)
        n_ratings_scaled=n_ratings_scaled.to(self.device)

        user_embedding=self.model.forward_user(items,
                                               techniques_users,
                                               n_items_scaled,
                                               ratings_scaled,
                                               n_ratings_scaled)
        
        user_embedding=user_embedding.reshape(1,-1)
        cos_sim_scores=torch.nn.functional.cosine_similarity(user_embedding,self.recipes_embeddings,dim=1).to(torch.float32)
        cos_sim_scores=pd.Series(cos_sim_scores.cpu().numpy())
        recipes_df["cos_sim_scores"]=cos_sim_scores
        recipes_df=recipes_df.drop_duplicates(subset="recipe_id")
        recipes_sorted=recipes_df.sort_values(by="cos_sim_scores",ascending=False)
        recipes_sorted=recipes_sorted[["recipe_id","i","name","description","tags","calorie_level","minutes","n_ingredients","ingredient_ids","ingredient_ids_continuous","nutrition","n_steps","steps"]]
        recommended_recipes=recipes_sorted.nlargest(n=n_recommended_recipes,columns=["cos_sim_scores"])

        return recommended_recipes,recipes_sorted



    def preprocessing(self,
                      items,
                      techniques_users,
                      n_items,
                      ratings,
                      n_ratings):
        
        if items is not None and not isinstance(items,list):
            items=list(map(int,ast.literal_eval(items)))
            true_len_items=len(items)
            items=list(np.pad(items,pad_width=(0,self.max_len_items-len(items))))
        elif items is None:
            items=[0]*self.max_len_items
            true_len_items=0
        else:
            true_len_items=len(items)
            items=list(np.pad(items,pad_width=(0,self.max_len_items-len(items))))

        if techniques_users is not None and not isinstance(techniques_users,list):
            techniques_users=list(map(int,ast.literal_eval(techniques_users)))
        elif techniques_users is None:
            techniques_users=[0]*self.n_techniques_users
        
        if ratings is not None and not isinstance(ratings,list):
            ratings=list(map(int,ast.literal_eval(ratings)))
            true_len_ratings=len(ratings)
            ratings=list(np.pad(ratings,pad_width=(0,self.max_len_items-len(ratings))))
        elif ratings is None:
            ratings=[0]*self.max_len_items
            true_len_ratings=0
        else:
            true_len_ratings=len(ratings)
            ratings=list(np.pad(ratings,pad_width=(0,self.max_len_items-len(ratings))))

        if n_items is not None and not isinstance(n_items,int):
            n_items=int(n_items)
        elif n_items is None:
            n_items=true_len_items
        else:
            n_items=n_items

        if n_ratings is not None and not isinstance(n_ratings,int):
            n_ratings=int(n_ratings)
        elif n_ratings is None:
            n_ratings=true_len_ratings
        else:
            n_ratings=n_ratings
        
        scaled=self.scaler_user.transform(np.numpy([n_items,n_ratings]))
        n_items_scaled=scaled[0]
        n_ratings_scaled=scaled[1]

        ratings_scaled=list(map(lambda x: x/5, ratings))

        items=torch.tensor(items,dtype=torch.long()).unsqueeze(0)
        techniques_users=torch.tensor(techniques_users,dtype=torch.long).unsqueeze(0)
        n_items_scaled=torch.tensor(n_items_scaled,dtype=torch.long).unsqueeze(0)
        n_ratings_scaled=torch.tensor(n_ratings_scaled,dtype=torch.long).unsqueeze(0)
        ratings_scaled=torch.tensor(ratings_scaled,dtype=torch.float32).unsqueeze(0)

        return items,techniques_users,n_items_scaled,n_ratings_scaled,ratings_scaled
