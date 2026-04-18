import ast
import numpy as np


class InferencePreprocessingRecipes():
    def __init__(self,
                 train_df,
                 scaler_recipes,
                 max_len_ingredients,
                 ingredient_continuous_ids,
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
        self.ingredient_continuous_ids=ingredient_continuous_ids
        self.recipe_continuous_ids=recipe_continuous_ids
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
        
        recipe_id,i,name,description,tags,calorie_level,\
        n_ingredients,ingredient_ids,nutrition,n_steps,\
        steps,techniques_recipes,ingredient_ids_continuous,full_text=self.before_split_processing(recipe_id,i,name,description,tags,calorie_level,n_ingredients,ingredient_ids,nutrition,n_steps,steps,techniques_recipes)

        minutes=self.after_split_processing(minutes)

        calorie_level_scaled,minutes_scaled,n_steps_scaled,n_ingredients_scaled=self.scale(calorie_level,minutes,n_steps,n_ingredients)

        cls_embedding,mean_embedding=self.bert_embeddings(full_text)

        return recipe_id,i,techniques_recipes,calorie_level_scaled,minutes_scaled,nutrition,n_ingredients_scaled,ingredient_ids_continuous,n_steps_scaled,cls_embedding,mean_embedding
    
    def bert_embeddings(self,full_text):
        tokenized_full_text=self.tokenizer(full_text,return_tensors="pt",max_length=512,truncation=True,padding="max_length")
        for key in tokenized_full_text:
            tokenized_full_text[key]=tokenized_full_text[key].to(self.device)
        output=self.bert_model(input_ids=tokenized_full_text["input_ids"],attention_mask=tokenized_full_text["attention_mask"]).last_hidden_state
        cls_embedding=output[:,0]
        mean_embedding=output.mean(dim=1)
        return cls_embedding,mean_embedding


    def scale(self,
              calorie_level,
              minutes,
              n_steps,
              n_ingredients):
        
        X=np.array([calorie_level,minutes,n_steps,n_ingredients],dtype=np.float32)
        X_scaled=self.scaler_recipes.transform(X)
        return X_scaled[0],X_scaled[1],X_scaled[2],X_scaled[3]



    def after_split_processing(self,minutes):
        if minutes is not None:
            minutes=int(minutes)
        else :
            minutes=self.train_df["minutes"].mode().iloc[0]
        return minutes

    def before_split_processing(self,
                    recipe_id=None,
                    i=None,
                    name=None,
                    description=None,
                    tags=None,
                    calorie_level=None,
                    n_ingredients=None,
                    ingredient_ids=None,
                    nutrition=None,
                    n_steps=None,
                    steps=None, 
                    techniques_recipes=None):
        
        if recipe_id is not None and i is None:
            if recipe_id in self.recipe_continuous_ids.index:
                i=int(self.recipe_continuous_ids.loc[recipe_id])
            else:
                i=178267
        elif recipe_id is None and i is None:
            i=0
        else:
            i=int(i)

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
            tags=" [SEP] ".join(ast.literal_eval(tags).replace("-"," "))
        
        if calorie_level is not None:
            calorie_level=int(calorie_level)
        else:
            calorie_level=0

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
                if len(nutrition)>self.n_nutrition:
                    nutrition=nutrition[:self.n_nutrition]
                nutrition=np.pad(nutrition,pad_width=(0,self.n_nutrition-len(nutrition)))
        else:
            nutrition=[0]*self.n_nutrition

        if n_steps is None and steps is not None:
            n_steps=steps.count("[SEP]")
        else:
            n_steps=0
        
        if techniques_recipes is not None:
            if not isinstance(techniques_recipes,list):
                techniques_recipes=list(map(int, ast.literal_eval(techniques_recipes)))
                if len(techniques_recipes)>self.n_techniques_recipes:
                    techniques_recipes=techniques_recipes[:self.n_techniques_recipes]
                techniques_recipes=np.pad(techniques_recipes,pad_width=(0,self.self.n_techniques_recipes-len(techniques_recipes)))
            else:
                if len(techniques_recipes)>self.n_techniques_recipes:
                    techniques_recipes=techniques_recipes[:self.n_techniques_recipes]
                techniques_recipes=np.pad(techniques_recipes,pad_width=(0,self.self.n_techniques_recipes-len(techniques_recipes)))
        else:
            techniques_recipes=[0]*self.n_techniques_recipes
        
        ingredient_ids_continuous=[self.ingredient_continuous_ids[i] if i in self.ingredient_continuous_ids.index else 0 for i in ingredient_ids]

        full_text=name+" [SEP] "+description+" [SEP] "+steps+" [SEP] "+tags

        return recipe_id,i,name,description,tags,calorie_level,n_ingredients,ingredient_ids,nutrition,n_steps,steps,techniques_recipes,ingredient_ids_continuous,full_text



        

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
    
    def before_split_processing(self,
                                user_id,
                                rating,
                                ratings,
                                n_ratings,
                                items,
                                n_items,
                                techniques_users
                                ):
        
        if user_id is not None:
            user_id=int(user_id)
        else:
            user_id=0

        if ratings is not None:
            if not isinstance(ratings,list):
                ratings=list(map(int, ast.literal_eval(ratings)))
                true_len_ratings=len(ratings)
                if len(ratings)>self.max_len_items:
                    ratings=ratings[:self.max_len_items]
                ratings=list(np.pad(ratings,pad_width=(self.max_len_items-len(ratings))))
            else:
                ratings=list(map(int,ratings))
                true_len_ratings=len(ratings)
                if len(ratings)>self.max_len_items:
                    ratings=ratings[:self.max_len_items]
                ratings=list(np.pad(ratings,pad_width=(self.max_len_items-len(ratings))))
        else:
            ratings=[0]*self.max_len_items
            true_len_ratings=0
        
        if n_ratings is not None:
            n_ratings=int(n_ratings)
        else:
            n_ratings=true_len_ratings
        
        if techniques_users is not None:
            if not isinstance(techniques_users,list):
                techniques_users=list(map(int, ast.literal_eval(techniques_users)))
                techniques_users=len(techniques_users)
                if len(techniques_users)>self.n_techniques_users:
                    techniques_users=techniques_users[:self.n_techniques_users]
                techniques_users=list(np.pad(techniques_users,pad_width=(0,self.n_techniques_users-len(techniques_users))))
            else:
                techniques_users=list(map(int,techniques_users))
                if len(techniques_users)>self.n_techniques_users:
                    techniques_users=techniques_users[:self.n_techniques_users]
                techniques_users=list(np.pad(techniques_users,pad_width=(0,self.n_techniques_users-len(techniques_users))))
                techniques_users=len(techniques_users)
        else:
            techniques_users=[0]*self.n_techniques_users
            techniques_users=0
        
        if items is not None:
            if not isinstance(items,list):
                items=list(map(int, ast.literal_eval(items)))
                true_len_items=len(items)
                if len(items)>self.max_len_items:
                    items=items[:self.max_len_items]
                items=list(np.pad(ratings,pad_width=(self.max_len_items-len(items))))
            else:
                items=list(map(int,items))
                true_len_items=len(items)
                if len(ratings)>self.max_len_items:
                    ratings=ratings[:self.max_len_items]
                ratings=list(np.pad(ratings,pad_width=(self.max_len_items-len(items))))
        else:
            ratings=[0]*self.max_len_items
            true_len_items=0
        
        if n_items is not None:
            n_items=int(n_items)
        else:
            n_items=true_len_items
        

