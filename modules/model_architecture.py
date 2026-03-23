import torch
import torch.nn as nn
import mmh3



class RecommendationModel(nn.Module):
    def __init__(self,
                 hashed_ingredients_ids_encoded_embeddings,
                 ingredient_id_emb_dim,
                 hashed_recipes_ids_encoded_embeddings,
                 recipe_id_emb_dim
                 distilbert_model,
                 distilbert_dmodel
                 ):
        super().__init__()

        self.hashed_ingredients_ids_encoded_embeddings=hashed_ingredients_ids_encoded_embeddings
        self.hashed_recipes_ids_encoded_embeddings=hashed_recipes_ids_encoded_embeddings

        self.ingredient_id_emb_dim=ingredient_id_emb_dim
        self.recipe_id_emb_dim=recipe_id_emb_dim

        self.distilbert_model=self.distilbert_model
        self.distilbert_dmodel=self.distilbert_dmodel

        self.dhe_fnn_ingredient=nn.Sequential(
            nn.Linear(self.ingredient_id_emb_dim,self.ingredient_id_emb_dim//2),
            nn.Mish(),
            nn.LayerNorm(self.ingredient_id_emb_dim//2),
            nn.Linear(self.ingredient_id_emb_dim//2,self.ingredient_id_emb_dim)
        )

        self.projection_ingredient=nn.Linear(self.ingredient_id_emb_dim,self.ingredient_id_emb_dim//4)

        self.dhe_fnn_items=nn.Sequential(
            nn.Linear(self.recipe_id_emb_dim,self.recipe_id_emb_dim//2),
            nn.Mish(),
            nn.LayerNorm(self.recipe_id_emb_dim//2),
            nn.Linear(self.recipe_id_emb_dim//2,self.recipe_id_emb_dim)
        )

        self.projection_ingredient=nn.Linear(self.recipe_id_emb_dim,self.recipe_id_emb_dim//4)

    
    def forward(self,
                user_id,
                recipe_id,
                rating,
                i,
                technique_recipes,
                calorie_level,
                ingredient_ids,
                ingredient_ids_continuous,
                techniques_users,
                items,
                n_items,
                ratings,
                n_ratings,
                minutes,
                nutrition,
                n_ingredients
                input_ids_steps,
                attention_mask_steps,
                input_ids_names,
                attention_mask_names,
                input_ids_descriptions,
                attention_mask_descriptions,
                input_ids_tags,
                attention_mask_tags
                ):
        
        encoded_ingredient_ids=self.hashed_ingredients_ids_encoded_embeddings[ingredient_ids_continuous,:]
        encoded_items=self.hashed_recipes_ids_encoded_embeddings[items,:]
