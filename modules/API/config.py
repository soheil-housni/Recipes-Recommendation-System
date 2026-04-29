from ..inference import RecipesRecommender
from ..utils import load
import joblib
import torch
import mlflow
import pyarrow.parquet as pq
from pathlib import Path
from ..model import RecommendationModel
from mlflow.tracking import MlflowClient



def initialisation():
    BASE_DIR = Path(__file__).resolve().parent
    mlflow.set_tracking_uri("file:./mlflow_runs/mlruns")
    client=MlflowClient()

    train_parquet_file=pq.ParquetFile("./modules/dataframes/train_df.parquet")
    train_df=load(train_parquet_file)

    recipes_embeddings=torch.load("./modules/recipes_set/recipes_embeddings.pt")

    recipes_parquet_file=pq.ParquetFile("./modules/recipes_set/recipes_df.parquet")
    recipes_df=load(recipes_parquet_file)

    scaler_user=joblib.load("./modules/scalers/scaler_users.pkl")

    hashed_ingredients_ids_encoded_embeddings=torch.load("./modules/hashed_encoded_tables/hashed_embeddings_ingredients.pt")
    hashed_recipes_ids_encoded_embeddings=torch.load("./modules/hashed_encoded_tables/hashed_embeddings_recipes.pt")


    device = (torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    """
    hp={}
    with open("./train_savings/model_5/hp.txt","r") as f:
        for line in f:
            line=line.strip().split(":")

            hp[line[0].strip()]=line[1].strip()
    
    dropout=float(hp["dropout"])
    projec_dropout=float(hp["projec_dropout"])
    mean_mode=bool(hp["mean_mode"])
    """
    registered_model_version=client.get_model_version("Best_Recommendation_System_model",1)

    dropout=float(registered_model_version.params["dropout"])
    projec_dropout=float(registered_model_version.params["projec_dropout"])
    mean_mode=bool(registered_model_version.params["mean_mode"])

    best_recommendation_model=RecommendationModel(
        hashed_ingredients_ids_encoded_embeddings=hashed_ingredients_ids_encoded_embeddings,
        hashed_recipes_ids_encoded_embeddings=hashed_recipes_ids_encoded_embeddings,
        device=device,
        dropout=dropout,
        projec_dropout=projec_dropout,
        mean_mode=mean_mode
    )

    run_id=registered_model_version.run_id
    print(run_id)
    run_name=client.get_run(f"{run_id}").info.run_name

    state_dict=torch.load(f"./train_savings/{run_name}/model.pt",map_location=device)
    best_recommendation_model.load_state_dict(state_dict)

    best_recommendation_model.device=device

    max_len_items=6437
    n_techniques_users=58

    return train_df,recipes_embeddings,recipes_df,scaler_user,device,best_recommendation_model,max_len_items,n_techniques_users