import pandas as pd
class Scaler():
    def __init__(self,scaler):
        self.scaler=scaler

    def scale(self,df:pd.DataFrame):
        scaler=self.scaler
        cols_to_scale=["calorie_level",'n_items','n_ratings',"minutes","n_steps","n_ingredients"]
        X_scaled=scaler.fit_transform(df[cols_to_scale])
        scaled_columns_df=pd.DataFrame({cols_to_scale[i]+"_scaled":X_scaled[:,i] for i in range(len(cols_to_scale))},index=df.index)
        df=pd.concat([df,scaled_columns_df],axis=1)
        return df
        