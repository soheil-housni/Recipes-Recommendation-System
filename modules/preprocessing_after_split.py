import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import StandardScaler



class AfterSplitPreprocessingUsersFeatures:

    def __init__(self,train_df):
        self.train_df_processed=train_df.copy()
        self.agg_user_ratings=self.train_df_processed.groupby("user_id")["rating"].agg(["count",lambda x: x.mode().iloc[0]]).rename(columns={'<lambda_0>':"mode"})
        self.agg_recipe_ratings=self.train_df_processed.groupby("recipe_id")["rating"].agg(["count",lambda x: x.mode().iloc[0]]).rename(columns={'<lambda_0>':"mode"})
        self.mode_rating=self.train_df_processed["rating"].mode().iloc[0]

    def preprocessing(self,df):
        df=self.filling_missing_values(df)
        return df


    def filling_missing_values(self,df):
        df["n_ratings"]=df["n_ratings"].fillna(df["user_id"].map(self.agg_user_ratings["count"])).fillna(0)
        df["rating"]=df["rating"].fillna(df["user_id"].map(self.agg_user_ratings["mode"])).fillna(df["recipe_id"].map(self.agg_recipe_ratings["mode"])).fillna(self.train_df_processed["rating"].mode().iloc[0]).fillna(0)
        return df




class AfterSplitPreprocessingRecipesFeatures:

    def __init__(self,train_df):
        self.train_df_processed=train_df.copy()
        self.agg_user_ratings=self.train_df_processed.groupby("user_id")["rating"].agg(["count",lambda x: x.mode().iloc[0]]).rename(columns={'<lambda_0>':"mode"})
        self.agg_recipe_ratings=self.train_df_processed.groupby("recipe_id")["rating"].agg(["count",lambda x: x.mode().iloc[0]]).rename(columns={'<lambda_0>':"mode"})
        self.mode_rating=self.train_df_processed["rating"].mode().iloc[0]

    def preprocessing(self,df):
        df=self.filling_missing_values(df)
        return df


    def filling_missing_values(self,df):
        df["minutes"]=df["minutes"].fillna(self.train_df_processed["minutes"].mode().iloc[0])
        return df