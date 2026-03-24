import torch
import torch.nn as nn
import mmh3



class RecommendationModel(nn.Module):
    def __init__(self,
                 hashed_ingredients_ids_encoded_embeddings,
                 ingredient_id_emb_dim,
                 hashed_recipes_ids_encoded_embeddings,
                 recipe_id_emb_dim,
                 distilbert_model,
                 distilbert_dmodel,
                 dropout
                 ):
        super().__init__()

        self.dropout=dropout

        self.hashed_ingredients_ids_encoded_embeddings=hashed_ingredients_ids_encoded_embeddings
        self.hashed_recipes_ids_encoded_embeddings=hashed_recipes_ids_encoded_embeddings

        self.ingredient_id_emb_dim=ingredient_id_emb_dim
        self.recipe_id_emb_dim=recipe_id_emb_dim

        self.distilbert_model=distilbert_model
        self.distilbert_dmodel=distilbert_dmodel

        self.dhe_fnn_ingredient=nn.Sequential(
            nn.Linear(self.ingredient_id_emb_dim,self.ingredient_id_emb_dim//2),
            nn.Mish(),
            nn.LayerNorm(self.ingredient_id_emb_dim//2),
            nn.Linear(self.ingredient_id_emb_dim//2,self.ingredient_id_emb_dim)
        )

        self.projection_ingredient=nn.Linear(self.ingredient_id_emb_dim,self.ingredient_id_emb_dim//4)
        self.norm_encoded_ingredients=nn.LayerNorm(self.ingredient_id_emb_dim//4)

        self.dhe_fnn_items=nn.Sequential(
            nn.Linear(self.recipe_id_emb_dim,self.recipe_id_emb_dim//2),
            nn.Mish(),
            nn.LayerNorm(self.recipe_id_emb_dim//2),
            nn.Linear(self.recipe_id_emb_dim//2,self.recipe_id_emb_dim)
        )

        self.projection_items=nn.Linear(self.recipe_id_emb_dim,self.recipe_id_emb_dim//4)
        self.norm_encoded_items=nn.LayerNorm(self.recipe_id_emb_dim//4)

        """

        self.projection_names=nn.Linear(self.distilbert_dmodel,self.distilbert_dmodel//4)
        self.norm_names=nn.LayerNorm(self.distilbert_dmodel//4)

        self.projection_steps=nn.Linear(self.distilbert_dmodel,self.distilbert_dmodel//4)
        self.norm_steps=nn.LayerNorm(self.distilbert_dmodel//4)

        self.projection_tags=nn.Linear(self.distilbert_dmodel,self.distilbert_dmodel//4)
        self.norm_tags=nn.LayerNorm(self.distilbert_dmodel//4)

        self.projection_descriptions=nn.Linear(self.distilbert_dmodel,self.distilbert_dmodel//4)
        self.norm_descriptions=nn.LayerNorm(self.distilbert_dmodel//4)

        """
        self.projection_full_text=nn.Linear(self.distilbert_dmodel,self.distilbert_dmodel//2)
        self.norm_full_text=nn.LayerNorm(self.distilbert_dmodel//2)
        self.projection_add_features_recipes=nn.Linear(3,3*4)
        self.norm_add_features_recipes=nn.LayerNorm(3*4)

        self.projection_add_features_users=nn.Linear(2,2*4)
        self.norm_add_features_users=nn.LayerNorm(2*4)

        self.projection_nutrition=nn.Linear(7,7*2)
        self.norm_nutrition=nn.LayerNorm(7*2)

        self.projection_techniques_users=nn.Linear(58,58)
        self.norm_techniques_users=nn.LayerNorm(58)

        self.projection_techniques_recipes=nn.Linear(58,58)
        self.norm_techniques_recipes=nn.LayerNorm(58)

        self.enter_dim_recipes=(3*4)+(7*2)+(58)+(self.distilbert_dmodel//2)+(recipe_id_emb_dim//4)
        self.enter_dim_users=(2*4)+58+(1024//4)

        self.user_fnn=nn.Sequential(
             nn.LayerNorm(self.enter_dim_users),
             nn.Linear(self.enter_dim_users,self.enter_dim_users//2),
             nn.LayerNorm(self.enter_dim_users//2),
             nn.ReLU(),
             nn.Dropout(self.dropout),
             nn.Linear(self.enter_dim_users//2,256),
             nn.LayerNorm(256)
        )

        self.recipe_fnn=nn.Sequential(
             nn.LayerNorm(self.enter_dim_recipes),
             nn.Linear(self.enter_dim_recipes,self.enter_dim_recipes//2),
             nn.LayerNorm(self.enter_dim_recipes//2),
             nn.ReLU(),
             nn.Dropout(self.dropout),
             nn.Linear(self.enter_dim_recipes//2,26),
             nn.LayerNorm(256)
        )

    """
    def weighted_mean(self,raw_ids,hashed_encoded_ids):
         mask=(raw_ids!=0).float()
         mask=mask.unsqueeze(dim=1)
         final_encoded=torch.matmul(mask,hashed_encoded_ids)
         final_encoded=final_encoded.squeeze(dim=1)
         mask_sum=mask.sum(dim=2).unsqueeze(dim=1)
         mask_sum=mask_sum.masked_fill((mask_sum==0),1.0)
         final_encoded=final_encoded/mask_sum
         return final_encoded
    """
    def weighted_mean_ingredients(self,raw_ids,hashed_encoded_ids):
         mask=(raw_ids!=0).float()
         mask=mask.unsqueeze(dim=1)
         final_encoded=torch.matmul(mask,hashed_encoded_ids)
         final_encoded=final_encoded.squeeze(dim=1)
         mask_sum=mask.sum(dim=2).unsqueeze(dim=1)
         mask_sum=mask_sum.masked_fill((mask_sum==0),1.0)
         final_encoded=final_encoded/mask_sum
         return final_encoded
    
    def weighted_mean_items(self,raw_ids,hashed_encoded_ids,ratings):
         mask=(raw_ids!=0).float()
         ratings=ratings.float()
         ratings=ratings*mask
         ratings=ratings.unsqueeze(dim=1).float()
         final_encoded=torch.matmul(ratings,hashed_encoded_ids).squeeze(1)
         ratings_sum=ratings.sum(dim=2).unsqueeze(dim=1)
         ratings_sum=ratings_sum.masked_fill((ratings_sum==0),1.0)
         final_encoded=final_encoded/ratings_sum
         return final_encoded


    
    def forward(self,
                user_id,
                recipe_id,
                rating,
                i,
                technique_recipes,
                calorie_level, #recipe
                ingredient_ids,
                ingredient_ids_continuous,
                techniques_users,
                items,
                n_items, #user
                ratings,
                n_ratings, #user
                minutes, #recipe
                nutrition, #recipe
                n_ingredients, #recipe
                input_ids_steps,
                attention_mask_steps,
                input_ids_names,
                attention_mask_names,
                input_ids_descriptions,
                attention_mask_descriptions,
                input_ids_tags,
                attention_mask_tags,
                input_ids_full,
                attention_mask_full
                ):
          
          encoded_ingredient_ids=self.hashed_ingredients_ids_encoded_embeddings[ingredient_ids_continuous,:]
          encoded_ingredient_ids=self.dhe_fnn_ingredient(encoded_ingredient_ids)
          encoded_ingredients_used=self.weighted_mean_ingredients(ingredient_ids_continuous,encoded_ingredient_ids)
          projected_encoded_ingredients=self.projection_ingredient(encoded_ingredients_used)
          projected_encoded_ingredients=self.norm_encoded_ingredients(projected_encoded_ingredients)

          encoded_items=self.hashed_recipes_ids_encoded_embeddings[items,:]
          encoded_items=self.dhe_fnn_items(encoded_items)
          encoded_items_history=self.weighted_mean(items,encoded_items)
          projected_encoded_items=self.projection_items(encoded_items_history)
          projected_encoded_items=self.norm_encoded_items(projected_encoded_items)
          

          """
          steps_embeddings=self.distilbert_model(input_ids=input_ids_steps,attention_mask=attention_mask_steps).last_hidden_state
          descriptions_embeddings=self.distilbert_model(input_ids=input_ids_descriptions,attention_mask=attention_mask_descriptions).last_hidden_state
          tags_embeddings=self.distilbert_model(input_ids=input_ids_tags,attention_mask=attention_mask_tags).last_hidden_state
          tags_names=self.distilbert_model(input_ids=input_ids_names,attention_mask=attention_mask_names).last_hidden_state

          pooled_steps=steps_embeddings.mean(dim=1)
          pooled_descriptions=descriptions_embeddings.mean(dim=1)
          pooled_tags=tags_embeddings.mean(dim=1)
          pooled_names=tags_names.mean(dim=1)

          projected_pooled_steps=self.projection_steps(pooled_steps)
          projected_pooled_steps=self.norm_steps(projected_pooled_steps)

          projected_pooled_names=self.projection_names(pooled_names)
          projected_pooled_names=self.norm_names(projected_pooled_names)

          projected_pooled_tags=self.projection_tags(pooled_tags)
          projected_pooled_tags=self.norm_tags(projected_pooled_tags)

          projected_pooled_descriptions=self.projection_descriptions(pooled_descriptions)
          projected_pooled_descriptions=self.norm_descriptions(projected_pooled_descriptions)

          """

          full_text=self.distilbert_model(input_ids=input_ids_full,attentin_mask=attention_mask_full).last_hidden_state
          pooled_full_text=full_text[:,0]
          projected_full_text=self.projection_full_text(pooled_full_text)
          projected_full_text=self.norm_full_text(projected_full_text)

          concat_add_features_users=torch.cat([n_items,n_ratings],dim=1)
          concat_add_features_users=self.norm_add_features_users(concat_add_features_users)

          concat_add_features_recipes=torch.cat([minutes,n_ingredients,calorie_level],dim=1)
          concat_add_features_recipes=self.norm_add_features_recipes(concat_add_features_recipes)

          concat_add_features_users=self.projection_add_features_users(concat_add_features_users)
          concat_add_features_recipes=self.projection_add_features_recipes(concat_add_features_recipes)

          nutrition=self.projection_nutrition(nutrition)
          nutrition=self.norm_nutrition(nutrition)

          technique_recipes=self.projection_techniques_recipes(technique_recipes)
          technique_recipes=self.norm_techniques_recipes(technique_recipes)

          techniques_users=self.projection_techniques_users(techniques_users)
          techniques_users=self.norm_techniques_users(techniques_users)


          user_embeddings=torch.cat([concat_add_features_users,techniques_users,projected_encoded_items],dim=1)
          recipe_embeddings=torch.cat([concat_add_features_recipes,nutrition,technique_recipes,projected_full_text,projected_encoded_ingredients],dim=1)

          user_embeddings=self.user_fnn(user_embeddings)
          recipe_embeddings=self.recipe_fnn(recipe_embeddings)

          return user_embeddings,recipe_embeddings











