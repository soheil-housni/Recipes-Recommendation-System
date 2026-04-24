import ast
import numpy as np
import torch


class InferencePreprocessingRecipes():
    def __init__(self,
                 train_df,
                 scaler_recipes,
                 max_len_ingredients,
                 ingredient_continuous_ids_serie,
                 recipe_continuous_ids,
                 n_nutrition,
                 n_techniques_recipes,
                 bert_model,
                 tokenizer,
                 device
                 ):
        
        self.scaler_recipes=scaler_recipes
        self.max_len_ingredients=max_len_ingredients
        self.train_df=train_df.copy()
        self.ingredient_continuous_ids_serie=ingredient_continuous_ids_serie
        self.recipe_continuous_ids_serie=recipe_continuous_ids
        self.n_nutrition=n_nutrition
        self.n_techniques_recipes=n_techniques_recipes
        self.bert_model=bert_model
        self.tokenizer=tokenizer
        self.device=device
        self.bert_model.to(self.device)

    def preprocessing(self,
                      recipe_id=None,
                      i=None,
                      name=None,
                      description=None,
                      tags=None,
                      calorie_level=None,
                      minutes=None,
                      n_ingredients=None,
                      ingredient_ids=None,
                      nutrition=None,
                      n_steps=None,
                      steps=None, 
                      techniques_recipes=None):
        
        recipe_id,i,name,description,tags,n_ingredients,ingredient_ids,\
        nutrition,n_steps,steps,techniques_recipes,ingredient_ids_continuous,full_text\
        =self.before_split_processing(recipe_id=recipe_id,
                                      i=i,
                                      name=name,
                                      description=description,
                                      tags=tags,
                                      n_ingredients=n_ingredients,
                                      ingredient_ids=ingredient_ids,
                                      nutrition=nutrition,
                                      n_steps=n_steps,
                                      steps=steps,
                                      techniques_recipes=techniques_recipes
                                      )        

        calorie_level,minutes=self.after_split_preprocessing(calorie_level=calorie_level,minutes=minutes)

        calorie_level_scaled,minutes_scaled,n_steps_scaled,n_ingredients_scaled=self.scale(calorie_level=calorie_level,
                                                                                           minutes=minutes,
                                                                                           n_steps=n_steps,
                                                                                           n_ingredients=n_ingredients)

        cls_embedding,mean_embedding=self.bert_embeddings(full_text)

        techniques_recipes=torch.tensor(techniques_recipes).reshape(1,-1)
        calorie_level_scaled=torch.tensor([calorie_level_scaled]).reshape(1,-1)
        ingredient_ids_continuous=torch.tensor(ingredient_ids_continuous).reshape(1,-1)
        minutes_scaled=torch.tensor([minutes_scaled]).reshape(1,-1)
        nutrition=torch.tensor(nutrition).reshape(1,-1)
        n_ingredients_scaled=torch.tensor([n_ingredients_scaled]).reshape(1,-1)
        n_steps_scaled=torch.tensor([n_steps_scaled]).reshape(1,-1)

        cls_embedding=cls_embedding.reshape(1,-1)
        mean_embedding=mean_embedding.reshape(1,-1)


        return techniques_recipes,calorie_level_scaled,ingredient_ids_continuous,minutes_scaled,nutrition,n_ingredients_scaled,n_steps_scaled,cls_embedding,mean_embedding
    
    def bert_embeddings(self,
                        full_text):
        with torch.inference_mode:
            self.bert_model.eval()
            tokenized_full_text=self.tokenizer(full_text,return_tensors="pt",max_length=512,truncation=True,padding="max_length")
            for key in tokenized_full_text:
                tokenized_full_text[key]=tokenized_full_text[key].to(self.device)
            attention_mask=tokenized_full_text["attention_mask"]
            output=self.bert_model(input_ids=tokenized_full_text["input_ids"],attention_mask=attention_mask).last_hidden_state
            cls_embedding=output[:,0]
            attention_mask=attention_mask.unsqueeze(-1).float()
            mask_sum=attention_mask.sum(dim=1).clamp(1.0).item()
            mean_embedding=(output*attention_mask).sum(dim=1)/mask_sum
        return cls_embedding,mean_embedding


    def scale(self,
              calorie_level,
              minutes,
              n_steps,
              n_ingredients):
        
        X=np.array([calorie_level,minutes,n_steps,n_ingredients],dtype=np.float32).reshape(1,-1)
        X_scaled=self.scaler_recipes.transform(X)
        calorie_level_scaled=X_scaled[0,0]
        minutes_scaled=X_scaled[0,1]
        n_steps_scaled=X_scaled[0,2]
        n_ingredients_scaled=X_scaled[0,3]
        return calorie_level_scaled,minutes_scaled,n_steps_scaled,n_ingredients_scaled



    def after_split_preprocessing(self,
                                  calorie_level=None,
                                  minutes=None):
        
        if calorie_level is not None:
            calorie_level=int(calorie_level)
        else:
            if self.train_df["calorie_level"].mode().iloc[0] is not None:
                calorie_level=self.train_df["calorie_level"].mode().iloc[0]
            else:
                calorie_level=0

        if minutes is not None:
            minutes=int(minutes)
        else :
            if self.train_df["minutes"].mode().iloc[0] is not None:
                minutes=self.train_df["minutes"].mode().iloc[0]
            else:
                minutes=0
        
        
        return calorie_level,minutes

    def before_split_processing(
                    self,
                    recipe_id=None,
                    i=None,
                    name=None,
                    description=None,
                    tags=None,
                    n_ingredients=None,
                    ingredient_ids=None,
                    nutrition=None,
                    n_steps=None,
                    steps=None, 
                    techniques_recipes=None
                    ):
        
        if recipe_id is not None and i is None:
            if recipe_id in self.recipe_continuous_ids_serie.index:
                i=int(self.recipe_continuous_ids_serie.loc[recipe_id])
            else:
                i=178266
        elif recipe_id is None and i is None:
            i=0
        else:
            i=int(i)
        
        if recipe_id is None:
            recipe_id=0
        else:
            recipe_id=(recipe_id)

        if name is None:
            name="no name"

        if description is None:
            description="missing description"
        
        if steps is None:
            steps="missing steps"
        else:
            steps=" [SEP] ".join(ast.literal_eval(steps))
        
        if tags is None:
            tags="no tags"
        else:
            tags=" [SEP] ".join(ast.literal_eval(tags)).replace("-"," ")
        

        if ingredient_ids is not None:
            if not isinstance(ingredient_ids,list):
                ingredient_ids=list(map(int,ast.literal_eval(ingredient_ids)))
                if len(ingredient_ids)>self.max_len_ingredients:
                    ingredient_ids=ingredient_ids[:self.max_len_ingredients]
                true_len_ingredients=len(ingredient_ids)
                ingredient_ids=np.pad(ingredient_ids,pad_width=(0,self.max_len_ingredients-len(ingredient_ids)))
            else:
                if len(ingredient_ids)>self.max_len_ingredients:
                    ingredient_ids=ingredient_ids[:self.max_len_ingredients]
                true_len_ingredients=len(ingredient_ids)
                ingredient_ids=np.pad(ingredient_ids,pad_width=(0,self.max_len_ingredients-len(ingredient_ids)))
        else:
            true_len_ingredients=0
            ingredient_ids=[0]*self.max_len_ingredients
        
        if n_ingredients is not None:
            n_ingredients=int(n_ingredients)
        else:
            n_ingredients=true_len_ingredients
        
        if nutrition is not None:
            if not isinstance(nutrition,list):
                nutrition=list(map(float,ast.literal_eval(nutrition)))
                if len(nutrition)>self.n_nutrition:
                    nutrition=nutrition[:self.n_nutrition]
                nutrition=np.pad(nutrition,pad_width=(0,self.n_nutrition-len(nutrition)))
            else:
                nutrition=list(map(float,nutrition))
                if len(nutrition)>self.n_nutrition:
                    nutrition=nutrition[:self.n_nutrition]
                nutrition=np.pad(nutrition,pad_width=(0,self.n_nutrition-len(nutrition)))
        else:
            nutrition=[0]*self.n_nutrition

        if n_steps is None:
            if steps is not None:
                n_steps=steps.count("[SEP]")
            else:
                n_steps=0
        else:
            n_steps=int(n_steps)
        
        if techniques_recipes is not None:
            if not isinstance(techniques_recipes,list):
                techniques_recipes=list(map(int, ast.literal_eval(techniques_recipes)))
                if len(techniques_recipes)>self.n_techniques_recipes:
                    techniques_recipes=techniques_recipes[:self.n_techniques_recipes]
                techniques_recipes=np.pad(techniques_recipes,pad_width=(0,self.n_techniques_recipes-len(techniques_recipes)))
            else:
                techniques_recipes=list(map(int,techniques_recipes))
                if len(techniques_recipes)>self.n_techniques_recipes:
                    techniques_recipes=techniques_recipes[:self.n_techniques_recipes]
                techniques_recipes=np.pad(techniques_recipes,pad_width=(0,self.n_techniques_recipes-len(techniques_recipes)))
        else:
            techniques_recipes=[0.0]*self.n_techniques_recipes
        
        ingredient_ids_continuous=[self.ingredient_continuous_ids_serie[i] if i in self.ingredient_continuous_ids_serie.index else 0 for i in ingredient_ids]

        full_text=name+" [SEP] "+description+" [SEP] "+steps+" [SEP] "+tags

        return recipe_id,i,name,description,tags,n_ingredients,ingredient_ids,nutrition,n_steps,steps,techniques_recipes,ingredient_ids_continuous,full_text



        

class InferencePreprocessingUsers():
    def __init__(self,
                 train_df,
                 max_len_items,
                 n_techniques_users,
                 scaler_user):
        
        self.train_df=train_df.copy()
        self.max_len_items=max_len_items
        self.scaler_user=scaler_user
        self.n_techniques_users=n_techniques_users

        self.agg_user_recipes=self.train_df.groupby("user_id")["recipe_id"].agg("count")
        self.agg_user_ratings=self.train_df.groupby("user_id")["rating"].agg(["count",lambda x: x.mode().iloc[0]]).rename(columns={'<lambda_0>':"mode"})
        self.mode_rating=self.train_df["rating"].mode().iloc[0]
    
    def preprocessing(self,
                      user_id=None,
                      ratings=None,
                      items=None,
                      techniques_users=None,
                      n_items=None,
                      n_ratings=None):
        
        user_id,ratings,items,techniques_users,ratings_scaled=self.before_split_processing(user_id=user_id,
                                                                                           ratings=ratings,
                                                                                           items=items,
                                                                                           techniques_users=techniques_users,
                                                                                           )
        
        n_items,n_ratings=self.after_split_preprocessing(user_id=user_id,
                                                         n_items=n_items,
                                                         n_ratings=n_ratings
                                                         )
        
        n_items_scaled,n_ratings_scaled=self.scale(n_items=n_items,
                                                   n_ratings=n_ratings)
        

        items=torch.tensor(items).reshape(1,-1).long()
        techniques_users=torch.tensor(techniques_users).reshape(1,-1).float()
        n_items_scaled=torch.tensor([n_items_scaled]).reshape(1,-1).float()
        ratings_scaled=torch.tensor(ratings_scaled).reshape(1,-1).float()
        n_ratings_scaled=torch.tensor([n_ratings_scaled]).reshape(1,-1).float()

        return items,techniques_users,n_items_scaled,ratings_scaled,n_ratings_scaled

    def scale(self,
              n_items,
              n_ratings):
        
        X=np.array([n_items,n_ratings], dtype=np.float32).reshape(1,-1)
        X_scaled=self.scaler_user.transform(X)
        n_items_scaled=X_scaled[0,0]
        n_ratings_scaled=X_scaled[0,1]

        return n_items_scaled,n_ratings_scaled
    
    def after_split_preprocessing(self,
                                  user_id,
                                  n_items=None,
                                  n_ratings=None,
                                  ):
        
        if n_items is not None:
            n_items=int(n_items)
        else:
            if user_id in self.agg_user_recipes.index and self.agg_user_recipes.loc[user_id] is not None:
                n_items=self.agg_user_recipes.loc[user_id]
            else:
                n_items=0

        if n_ratings is not None:
            n_ratings=int(n_ratings)
        else:
            if user_id in self.agg_user_ratings.index and self.agg_user_ratings.index.loc[user_id]["count"] is not None:
                n_ratings=self.agg_user_ratings.loc[user_id]["count"]
            else:
                n_ratings=0

        return n_items,n_ratings,
    
    def before_split_processing(self,
                                user_id=None,
                                ratings=None,
                                items=None,
                                techniques_users=None
                                ):
        
        if user_id is not None:
            user_id=int(user_id)
        else:
            user_id=0

        if ratings is not None:
            if not isinstance(ratings,list):
                ratings=list(map(int, ast.literal_eval(ratings)))
                if len(ratings)>self.max_len_items:
                    ratings=ratings[:self.max_len_items]
                ratings=list(np.pad(ratings,pad_width=(0,self.max_len_items-len(ratings))))
            else:
                if len(ratings)>self.max_len_items:
                    ratings=ratings[:self.max_len_items]
                ratings=list(np.pad(ratings,pad_width=(0,self.max_len_items-len(ratings))))
        else:
            ratings=[0]*self.max_len_items
        
        if techniques_users is not None:
            if not isinstance(techniques_users,list):
                techniques_users=list(map(int, ast.literal_eval(techniques_users)))
                if len(techniques_users)>self.n_techniques_users:
                    techniques_users=techniques_users[:self.n_techniques_users]
                techniques_users=list(np.pad(techniques_users,pad_width=(0,self.n_techniques_users-len(techniques_users))))
            else:
                if len(techniques_users)>self.n_techniques_users:
                    techniques_users=techniques_users[:self.n_techniques_users]
                techniques_users=list(np.pad(techniques_users,pad_width=(0,self.n_techniques_users-len(techniques_users))))
        else:
            techniques_users=[0]*self.n_techniques_users
        
        if items is not None:
            if not isinstance(items,list):
                items=list(map(int, ast.literal_eval(items)))
                if len(items)>self.max_len_items:
                    items=items[:self.max_len_items]
                items=list(np.pad(items,pad_width=(0,self.max_len_items-len(items))))
            else:
                if len(ratings)>self.max_len_items:
                    ratings=ratings[:self.max_len_items]
                items=list(np.pad(items,pad_width=(0,self.max_len_items-len(items))))
        else:
            items=[0]*self.max_len_items
        
        ratings_scaled=list(map(lambda x : x/5, ratings))

        
        return user_id,ratings,items,techniques_users,ratings_scaled

