import torch
import gc
from loguru import logger
import optuna
from .creation_dataset import CreationDataset
from .dataloader_collate_function import CollateFunction
from .model_architecture import RecommendationModel
from .control_seed import seed_worker
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from .train import Train
import os

class OptunaFunction():
    def __init__(self,
                 train_df,
                 val_df,
                 train_cls_embeddings,
                 val_cls_embeddings,
                 train_mean_embeddings,
                 val_mean_embeddings,
                 hashed_ingredients_ids_encoded_embeddings,
                 hashed_recipes_ids_encoded_embeddings,
                 loss_fn,
                 device,
                 tokenizer,
                 hyperparemeters_ranges,
                 seed:int=42):
        
        self.seed=seed
        self.train_df=train_df
        self.val_df=val_df

        self.train_cls_embeddings=train_cls_embeddings
        self.val_cls_embeddings=val_cls_embeddings

        self.train_mean_embeddings=train_mean_embeddings
        self.val_mean_embeddings=val_mean_embeddings

        self.hashed_ingredients_ids_encoded_embeddings=hashed_ingredients_ids_encoded_embeddings
        self.hashed_recipes_ids_encoded_embeddings=hashed_recipes_ids_encoded_embeddings

        self.loss_fn=loss_fn

        self.tokenizer=tokenizer

        self.device=device

        self.hyperparemeters_ranges=hyperparemeters_ranges
    
    def objective(self,
                  trial):
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        try:
            batch_size = trial.suggest_categorical("batch_size",self.hyperparemeters_ranges["batch_size"])
            dropout = trial.suggest_float("dropout",self.hyperparemeters_ranges["dropout"]["low"],self.hyperparemeters_ranges["dropout"]["high"],step=self.hyperparemeters_ranges["dropout"]["step"])
            projec_dropout= trial.suggest_float("projec_dropout",self.hyperparemeters_ranges["projec_dropout"]["low"],self.hyperparemeters_ranges["projec_dropout"]["high"],step=self.hyperparemeters_ranges["dropout"]["step"])
            lr=trial.suggest_float("lr",self.hyperparemeters_ranges["lr"]["low"],self.hyperparemeters_ranges["lr"]["high"],log=self.hyperparemeters_ranges["lr"]["log"])
            weight_decay=trial.suggest_float("weight_decay",self.hyperparemeters_ranges["weight_decay"]["low"],self.hyperparemeters_ranges["weight_decay"]["high"],log=self.hyperparemeters_ranges["weight_decay"]["log"])
            warmup_prop=trial.suggest_float("warmup_prop",self.hyperparemeters_ranges["warmup_prop"]["low"],self.hyperparemeters_ranges["warmup_prop"]["high"],step=self.hyperparemeters_ranges["warmup_prop"]["step"])
            mean_mode=trial.suggest_categorical("mean_mode",self.hyperparemeters_ranges["mean_mode"])

            train_dataset=CreationDataset(self.train_df,cls_embeddings=self.train_cls_embeddings,mean_embeddings=self.train_mean_embeddings)
            val_dataset=CreationDataset(self.val_df,cls_embeddings=self.val_cls_embeddings,mean_embeddings=self.val_mean_embeddings)
            
            model=RecommendationModel(hashed_ingredients_ids_encoded_embeddings=self.hashed_ingredients_ids_encoded_embeddings,hashed_recipes_ids_encoded_embeddings=self.hashed_recipes_ids_encoded_embeddings,device=self.device,dropout=dropout,projec_dropout=projec_dropout,mean=mean_mode)
            generator=torch.Generator()
            generator.manual_seed(self.seed)
            collate_function_object=CollateFunction(tokenizer=self.tokenizer)
            collate_fn=collate_function_object.collate_fn
            train_dataloader=DataLoader(train_dataset,batch_size=batch_size,collate_fn=collate_fn,generator=generator,worker_init_fn=seed_worker,shuffle=True,drop_last=True)
            val_dataloader=DataLoader(val_dataset,batch_size=batch_size,collate_fn=collate_fn,generator=generator,worker_init_fn=seed_worker,shuffle=True,drop_last=True)

            trainer=Train(train_dataloader=train_dataloader,val_dataloader=val_dataloader,model=model,device=self.device,loss_fn=self.loss_fn,warmup_prop=warmup_prop,lr=lr,weight_decay=weight_decay)

            folder_name=f"mean_mode_{mean_mode}_bs_{batch_size}_d_{dropout}_pd_{projec_dropout}_lr_{lr}_wd_{weight_decay}_wp_{warmup_prop}".replace(".",",")
            if not os.path.exists(f"./train_savings/{folder_name}"):
                os.makedirs(f"./train_savings/{folder_name}")
            path=f"./train_savings/{folder_name}"

            logger.info(f"Trial number {trial.number}: bs:{batch_size}_d:{dropout}_pd:{projec_dropout}_lr:{lr}_wd:{weight_decay}_wp:{warmup_prop}".replace(".",","))
            strict_best_total_loss=trainer.run_training(path=path,trial=trial)
            print("----------------------------------------------------------------------------------------------------")
            return strict_best_total_loss
        
        finally:
            del model
            del trainer
            del train_dataloader
            del val_dataloader
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            gc.collect()