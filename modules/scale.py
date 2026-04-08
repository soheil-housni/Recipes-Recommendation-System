import pandas as pd
class Scaler():
    def __init__(self,scaler):
        self.scaler=scaler
        self.cols_to_scale=["calorie_level",'n_items','n_ratings',"minutes","n_steps","n_ingredients"]

    def scale(self,
              train_df:pd.DataFrame,
              val_df:pd.DataFrame,
              test_df:pd.DataFrame):
        X_train_scaled=self.scaler.fit_transform(train_df[self.cols_to_scale])
        X_val_scaled=self.scaler.transform(val_df[self.cols_to_scale])
        X_test_scaled=self.scaler.transform(test_df[self.cols_to_scale])
        train_df=self.concatenation(train_df,X_train_scaled)
        val_df=self.concatenation(val_df,X_val_scaled)
        test_df=self.concatenation(test_df,X_test_scaled)
        return train_df,val_df,test_df
    
    def concatenation(self,df,X_scaled):
        scaled_columns_df=pd.DataFrame({self.cols_to_scale[i]+"_scaled":X_scaled[:,i] for i in range(len(self.cols_to_scale))},index=df.index)
        df=pd.concat([df,scaled_columns_df],axis=1)
        return df

