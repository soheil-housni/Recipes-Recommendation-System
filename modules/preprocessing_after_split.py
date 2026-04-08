import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import StandardScaler



class AfterSplitPreprocessing:

    def __init__(self,train_df,val_df,test_df):
        self.train_df_processed=train_df.copy()
        self.val_df_processed=val_df.copy()
        self.test_df_processed=test_df.copy()

        self.agg_user_ratings=self.train_df_processed.groupby("user_id")["rating"].agg(["count",lambda x: x.mode().iloc[0]]).rename(columns={'<lambda_0>':"mode"})
        self.agg_recipe_ratings=self.train_df_processed.groupby("recipe_id")["rating"].agg(["count",lambda x: x.mode().iloc[0]]).rename(columns={'<lambda_0>':"mode"})
        self.mode_rating=self.train_df_processed["rating"].mode().iloc[0]

    def preprocessing(self):
        self.filling_missing_values()
        return self.train_df_processed,self.val_df_processed,self.test_df_processed

    def filling_missing_values(self):
        self.train_df_processed["minutes"]=self.train_df_processed["minutes"].fillna(self.train_df_processed["minutes"].mode().iloc[0])
        self.val_df_processed["minutes"]=self.val_df_processed["minutes"].fillna(self.train_df_processed["minutes"].mode().iloc[0])
        self.test_df_processed["minutes"]=self.test_df_processed["minutes"].fillna(self.train_df_processed["minutes"].mode().iloc[0])

        self.train_df_processed["n_ratings"]=self.train_df_processed["n_ratings"].fillna(self.train_df_processed["user_id"].map(self.agg_user_ratings["count"])).fillna(0)
        self.val_df_processed["n_ratings"]=self.val_df_processed["n_ratings"].fillna(self.val_df_processed["user_id"].map(self.agg_user_ratings["count"])).fillna(0)
        self.test_df_processed["n_ratings"]=self.test_df_processed["n_ratings"].fillna(self.test_df_processed["user_id"].map(self.agg_user_ratings["count"])).fillna(0)

        self.train_df_processed["rating"]=self.train_df_processed["rating"].fillna(self.train_df_processed["user_id"].map(self.agg_user_ratings["mode"])).fillna(self.train_df_processed["recipe_id"].map(self.agg_recipe_ratings["mode"])).fillna(self.train_df_processed["rating"].mode().iloc[0]).fillna(0)
        self.val_df_processed["rating"]=self.val_df_processed["rating"].fillna(self.val_df_processed["user_id"].map(self.agg_user_ratings["mode"])).fillna(self.val_df_processed["recipe_id"].map(self.agg_recipe_ratings["mode"])).fillna(self.train_df_processed["rating"].mode().iloc[0]).fillna(0)
        self.test_df_processed["rating"]=self.test_df_processed["rating"].fillna(self.test_df_processed["user_id"].map(self.agg_user_ratings["mode"])).fillna(self.test_df_processed["recipe_id"].map(self.agg_recipe_ratings["mode"])).fillna(self.train_df_processed["rating"].mode().iloc[0]).fillna(0)
    