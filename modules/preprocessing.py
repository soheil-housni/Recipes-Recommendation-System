import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import StandardScaler



class DataFramePreprocessing:

    def __init__(self,full_df,max_len_ingredients,max_len_items):
        self.full_df_processed=full_df.copy()


        self.max_len_ingredients=max_len_ingredients
        self.max_len_items=max_len_items

        self.agg_user_ratings=self.full_df_processed.groupby("user_id")["rating"].agg(["count",lambda x: x.mode().iloc[0]]).rename(columns={'<lambda_0>':"mode"})
        self.agg_recipe_ratings=self.full_df_processed.groupby("recipe_id")["rating"].agg(["count",lambda x: x.mode().iloc[0]]).rename(columns={'<lambda_0>':"mode"})
        self.mode_rating=self.full_df_processed["rating"].mode().iloc[0]


    def preprocessing(self):
        self.transforming()
        self.filling_missing_values()
        return self.full_df_processed


    def filling_missing_values(self):
        n_techniques_recipes=len(self.full_df_processed.loc[~self.full_df_processed["techniques_recipes"].isna(),"techniques_recipes"].iloc[0])
        n_techniques_users=len(self.full_df_processed.loc[~self.full_df_processed["techniques_users"].isna(),"techniques_users"].iloc[0])
        n_nutrition=len(self.full_df_processed.loc[~self.full_df_processed["nutrition"].isna(),"nutrition"].iloc[0])
        
        self.full_df_processed["steps"]=self.full_df_processed["steps"].fillna("missing steps")
        self.full_df_processed["description"]=self.full_df_processed["description"].fillna("missing description")
        self.full_df_processed["techniques_recipes"]=self.full_df_processed["techniques_recipes"].apply(lambda x : x if isinstance(x,list) else [0]*n_techniques_recipes)
        self.full_df_processed["techniques_users"]=self.full_df_processed["techniques_users"].apply(lambda x : x if isinstance(x,list) else [0]*n_techniques_users)
        self.full_df_processed["items"]=self.full_df_processed["items"].apply(lambda x : x if isinstance(x,list) else [0]*self.max_len_items)
        self.full_df_processed["n_items"]=self.full_df_processed["n_items"].fillna(0)
        self.full_df_processed["name"]=self.full_df_processed["name"].fillna("no name")
        self.full_df_processed["minutes"]=self.full_df_processed["minutes"].fillna(self.full_df_processed["minutes"].mode().iloc[0])
        self.full_df_processed["tags"]=self.full_df_processed["tags"].fillna("no tags")
        self.full_df_processed["nutrition"]=self.full_df_processed["nutrition"].apply(lambda x : x if isinstance(x,list) else [0]*n_nutrition)
        self.full_df_processed["i"]=self.full_df_processed["i"]+1

        self.full_df_processed["n_ratings"]=self.full_df_processed["n_ratings"].fillna(self.full_df_processed["user_id"].map(self.agg_user_ratings["count"])).fillna(0)

        mask_missing_n_steps=self.full_df_processed["n_steps"].isna()
        self.full_df_processed.loc[mask_missing_n_steps,"n_steps"]=self.full_df_processed.loc[mask_missing_n_steps,"steps"].str.count(r"\[SEP\]").fillna(0)

        mask_missing_n_ingredients=self.full_df_processed["n_ingredients"].isna()
        self.full_df_processed.loc[mask_missing_n_ingredients,"n_ingredients"]=self.full_df_processed.loc[mask_missing_n_ingredients,"ingredient_ids"].apply(lambda x: len(x) if isinstance(x,list) else 0)

        self.full_df_processed["rating"]=self.full_df_processed["rating"].fillna(self.full_df_processed["user_id"].map(self.agg_user_ratings["mode"])).fillna(self.full_df_processed["recipe_id"].map(self.agg_recipe_ratings["mode"])).fillna(self.full_df_processed["rating"].mode().iloc[0])

        mask_missing_n_items=self.full_df_processed["n_items"].isna()
        self.full_df_processed.loc[mask_missing_n_items,"n_items"]=self.full_df_processed.loc[mask_missing_n_items,"items"].apply(lambda x : len(x) if isinstance(x,list) else 0)


    def transforming(self):
        self.full_df_processed["user_id"]=self.full_df_processed["user_id"].astype(int)
        self.full_df_processed["recipe_id"]=self.full_df_processed["recipe_id"].astype(int)
        self.full_df_processed["rating"]=self.full_df_processed["rating"].astype(float)

        self.full_df_processed=self.full_df_processed.apply(self.process_row,axis=1)


        self.full_df_processed["items"]=self.full_df_processed["items"].apply(lambda x: list(np.pad(x,pad_width=(0,self.max_len_items-len(x)))))
        self.full_df_processed["ratings"]=self.full_df_processed["ratings"].apply(lambda x: list(np.pad(x,pad_width=(0,self.max_len_items-len(x)))))


        self.full_df_processed["ingredient_ids"]=self.full_df_processed["ingredient_ids"].apply(lambda x: list(np.pad(x,pad_width=(0,self.max_len_ingredients-len(x)))))

    def process_row(self,row):
        row["techniques_recipes"] = list(map(int, ast.literal_eval(row["techniques_recipes"])))
        row["ingredient_ids"] = list(map(int, ast.literal_eval(row["ingredient_ids"])))
        row["techniques_users"] = list(map(int, ast.literal_eval(row["techniques_users"])))
        row["items"] = list(map(lambda x: int(x)+1, ast.literal_eval(row["items"])))
        row["ratings"] = list(map(int, ast.literal_eval(row["ratings"])))
        row["nutrition"] = list(map(float, ast.literal_eval(row["nutrition"])))
        row["tags"] = " [SEP] ".join(ast.literal_eval(row["tags"].replace("-"," ")))
        row["steps"] = " [SEP] ".join(ast.literal_eval(row["steps"]))
        return row


    