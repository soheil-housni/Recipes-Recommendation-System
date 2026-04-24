import torch
import pandas as pd

def creation_recipes_set(df,cls_embeddings,mean_embeddings,path):
    recipes_df=df.copy()
    recipes_df=recipes_df[["recipe_id","i","name","description","tags","calorie_level","minutes","n_ingredients","ingredient_ids","nutrition","n_steps","steps","techniques_recipes"]]
    recipes_df=recipes_df.drop_duplicates(subset="i").sort_values(by="i")
    idx=list(recipes_df.index)
    cls_embeddings=cls_embeddings[idx]
    mean_embeddings=mean_embeddings[idx]

    recipes_df.to_parquet(f"{path}/recipes_df.parquet")
    torch.save(cls_embeddings,f"{path}/recipes_set_cls_embeddings.pt")
    torch.save(mean_embeddings,f"{path}/recipes_set_mean_embeddings.pt")
    return recipes_df,cls_embeddings,mean_embeddings





