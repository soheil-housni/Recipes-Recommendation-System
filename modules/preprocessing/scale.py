import pandas as pd
class Scaler():
    def __init__(self,scaler_users,scaler_recipes):
        self.scaler_users=scaler_users
        self.scaler_recipes=scaler_recipes
        self.cols_to_scale_users=['n_items','n_ratings']
        self.cols_to_scale_recipes=["calorie_level","minutes","n_steps","n_ingredients"]

    def scale(self,
              train_df:pd.DataFrame,
              val_df:pd.DataFrame,
              test_df:pd.DataFrame):
        
        X_train_scaled_users=self.scaler_users.fit_transform(train_df[self.cols_to_scale_users])
        X_val_scaled_users=self.scaler_users.transform(val_df[self.cols_to_scale_users])
        X_test_scaled_users=self.scaler_users.transform(test_df[self.cols_to_scale_users])

        X_train_scaled_recipes=self.scaler_recipes.fit_transform(train_df[self.cols_to_scale_recipes])
        X_val_scaled_recipes=self.scaler_recipes.transform(val_df[self.cols_to_scale_recipes])
        X_test_scaled_recipes=self.scaler_recipes.transform(test_df[self.cols_to_scale_recipes])

        train_df=self.concatenation(train_df,X_train_scaled_users,X_train_scaled_recipes)
        val_df=self.concatenation(val_df,X_val_scaled_users,X_val_scaled_recipes)
        test_df=self.concatenation(test_df,X_test_scaled_users,X_test_scaled_recipes)
        return train_df,val_df,test_df
    
    def concatenation(self,df,X_scaled_users,X_scaled_recipes):
        users_scaled_columns_df=pd.DataFrame({self.cols_to_scale_users[i]+"_scaled":X_scaled_users[:,i] for i in range(len(self.cols_to_scale_users))},index=df.index)
        recipes_scaled_columns_df=pd.DataFrame({self.cols_to_scale_recipes[i]+"_scaled":X_scaled_recipes[:,i] for i in range(len(self.cols_to_scale_recipes))},index=df.index)
        df=pd.concat([df,users_scaled_columns_df,recipes_scaled_columns_df],axis=1)
        return df

class ScalerInferenceUsers():
    def __init__(self,scaler_user):
        self.scaler_user=scaler_user
        self.cols_to_scale_users=['n_items','n_ratings']
    
    def scale(self,df):
        X_scaled_users=self.scaler_user.transform(df[self.cols_to_scale_users])
        df=self.concatenation(df,X_scaled_users)
        return df
    
    def concatenation(self,df,X_scaled_users):
        users_scaled_columns_df=pd.DataFrame({self.cols_to_scale_users[i]+"_scaled":X_scaled_users[:,i] for i in range(len(self.cols_to_scale_users))},index=df.index)
        df=pd.concat([df,users_scaled_columns_df],axis=1)
        return df

class ScalerInferenceRecipes():
    def __init__(self,scaler_recipes):
        self.scaler_recipes=scaler_recipes
        self.cols_to_scale_recipes=["calorie_level","minutes","n_steps","n_ingredients"]
    
    def scale(self,df):
        X_scaled_recipes=self.scaler_recipes.transform(df[self.cols_to_scale_recipes])
        df=self.concatenation(df,X_scaled_recipes)
        return df
    
    def concatenation(self,df,X_scaled_recipes):
        recipes_scaled_columns_df=pd.DataFrame({self.cols_to_scale_recipes[i]+"_scaled":X_scaled_recipes[:,i] for i in range(len(self.cols_to_scale_recipes))},index=df.index)
        df=pd.concat([df,recipes_scaled_columns_df],axis=1)
        return df

