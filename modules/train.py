import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np
from loguru import logger
import mlflow
import optuna

class Train():
    def __init__(self,
                 train_dataloader,
                 val_dataloader,
                 model,
                 device,
                 loss_fn,
                 warmup_prop:float=0.1,
                 lr:float=0.01,
                 weight_decay:float=5e-4,
                 n_epochs:int=10,
                 patience:int=2,
                 min_improvement=1e-4,
                 mean:bool=False
                 ):
        
        self.mean=mean
        self.train_dataloader=train_dataloader
        self.val_dataloader=val_dataloader
        self.model=model
        self.device=device
        self.lr=lr
        self.weight_decay=weight_decay
        self.criterion=loss_fn
        self.n_epochs=n_epochs
        self.patience=patience
        self.min_improvement=min_improvement
        self.warmup_prop=warmup_prop
        
        #distilbert_parameters,backbone_parameters=self.get_parameters()
        #self.optimizer=AdamW([{"params":distilbert_parameters,"lr":1e-5},{"params":backbone_parameters,"lr":self.lr}],weight_decay=self.weight_decay)
        self.optimizer=AdamW(params=self.model.parameters(),lr=self.lr,weight_decay=self.weight_decay)

        num_training_steps=len(train_dataloader)*n_epochs
        num_warmup_steps=self.warmup_prop*num_training_steps

        self.scheduler=get_linear_schedule_with_warmup(optimizer=self.optimizer,num_training_steps=num_training_steps,num_warmup_steps=num_warmup_steps)

        self.model.to(self.device)
        #self.freeze_layers()
    
    """
    def get_parameters(self):
        distilbert_parameters=[param for param in list(self.model.children())[0].parameters()]
        backbone_parameters=[parameter for module in list(self.model.children())[1:] for parameter in module.parameters()]
        return distilbert_parameters,backbone_parameters
    """

    """
    def freeze_layers(self):
        for param in self.model.distilbert_model.embeddings.parameters():
            param.requires_grad=False

        for i in range(self.n_frozen_layers):
            for param in self.model.distilbert_model.transformer.layer[i].parameters():
                param.requires_grad=False
    """


    def run_training(self,path:str,trial:None):

        train_epoch_total_losses=[]
        val_epoch_total_losses=[]

        train_epoch_contrastive_losses=[]
        val_epoch_contrastive_losses=[]

        train_epoch_mse_losses=[]
        val_epoch_mse_losses=[]

        best_model=None
        strict_best_total_loss=float("inf")
        best_total_loss=float("inf")
        best_contrastive_loss=float("inf")
        best_mse_loss=float("inf")
        counter=0

        for epoch in range(self.n_epochs):
            logger.info(f"Epoch {epoch} : ")

            self.model.train()

            train_batch_total_losses=[]
            train_batch_contrastive_losses=[]
            train_batch_mse_losses=[]

            val_batch_total_losses=[]
            val_batch_contrastive_losses=[]
            val_batch_mse_losses=[]

            for batch in self.train_dataloader:
                for key in list(batch.keys()):
                    batch[key]=batch[key].to(self.device)
                outputs=self.model(technique_recipes=batch["technique_recipes"],
                                   calorie_level_scaled=batch["calorie_level_scaled"],
                                   ingredient_ids_continuous=batch["ingredient_ids_continuous"],
                                   techniques_users=batch["techniques_users"],
                                   items=batch["items"],
                                   n_items_scaled=batch["n_items_scaled"],
                                   ratings_scaled=batch["ratings_scaled"],
                                   n_ratings_scaled=batch["n_ratings_scaled"],
                                   minutes_scaled=batch["minutes_scaled"],
                                   nutrition=batch["nutrition"],
                                   n_ingredients_scaled=batch["n_ingredients_scaled"],
                                   cls_embeddings=batch["cls_embeddings"],
                                   mean_embeddings=batch["mean_embeddings"]
                                   #input_ids_full=batch["input_ids_full"],
                                   #attention_mask_full=batch["attention_mask_full"]
                                   )
                
                self.optimizer.zero_grad()
                loss=self.criterion(
                    outputs=outputs,
                    rating_scaled=batch["rating_scaled"],
                )
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                train_batch_total_losses.append(loss.detach().item())
                train_batch_contrastive_losses.append(self.criterion.access_loss_components()["Contrastive loss"])
                train_batch_mse_losses.append(self.criterion.access_loss_components()["MSE loss"])
            
            self.model.eval()
            with torch.no_grad():
                for batch in self.val_dataloader:
                    for key in list(batch.keys()):
                        batch[key]=batch[key].to(self.device)
                    outputs=self.model(technique_recipes=batch["technique_recipes"],
                                   calorie_level_scaled=batch["calorie_level_scaled"],
                                   ingredient_ids_continuous=batch["ingredient_ids_continuous"],
                                   techniques_users=batch["techniques_users"],
                                   items=batch["items"],
                                   n_items_scaled=batch["n_items_scaled"],
                                   ratings_scaled=batch["ratings_scaled"],
                                   n_ratings_scaled=batch["n_ratings_scaled"],
                                   minutes_scaled=batch["minutes_scaled"],
                                   nutrition=batch["nutrition"],
                                   n_ingredients_scaled=batch["n_ingredients_scaled"],
                                   cls_embeddings=batch["cls_embeddings"],
                                   mean_embeddings=batch["mean_embeddings"]
                                   #input_ids_full=batch["input_ids_full"],
                                   #attention_mask_full=batch["attention_mask_full"]
                                   )
                    loss=self.criterion(
                    outputs=outputs,
                    rating_scaled=batch["rating_scaled"],
                    )

                    val_batch_total_losses.append(loss.detach().item())
                    val_batch_contrastive_losses.append(self.criterion.access_loss_components()["Contrastive loss"])
                    val_batch_mse_losses.append(self.criterion.access_loss_components()["MSE loss"])
            
            
            train_total_loss=np.mean(train_batch_total_losses)
            train_epoch_total_losses.append(train_total_loss)
            train_contrastive_loss=np.mean(train_batch_contrastive_losses)
            train_epoch_contrastive_losses.append(train_contrastive_loss)
            train_mse_loss=np.mean(train_batch_mse_losses)
            train_epoch_mse_losses.append(train_mse_loss)

            val_total_loss=np.mean(val_batch_total_losses)
            val_epoch_total_losses.append(val_total_loss)
            val_contrastive_loss=np.mean(val_batch_contrastive_losses)
            val_epoch_contrastive_losses.append(val_contrastive_loss)
            val_mse_loss=np.mean(val_batch_mse_losses)
            val_epoch_mse_losses.append(val_mse_loss)


            
            logger.info(f"Epoch {epoch} : train total loss = {train_total_loss}")
            logger.info(f"Epoch {epoch} : train contrative loss = {train_contrastive_loss}")
            logger.info(f"Epoch {epoch} : train MSE loss = {train_mse_loss}")

            logger.info(f"Epoch {epoch} : validation total loss = {val_total_loss}")
            logger.info(f"Epoch {epoch} : validation contrastive loss = {val_contrastive_loss}")
            logger.info(f"Epoch {epoch} : validation MSE loss = {val_mse_loss}")

            if val_total_loss<=best_total_loss-self.min_improvement:
                best_total_loss=val_total_loss
                strict_best_total_loss=best_total_loss
                best_contrastive_loss=val_contrastive_loss
                best_mse_loss=val_mse_loss
                counter=0
                best_model=self.model
                torch.save(best_model.state_dict(),f"{path}/model.pt")
            else:
                if val_total_loss<=best_total_loss:
                    strict_best_total_loss=val_total_loss
                    best_contrastive_loss=val_contrastive_loss
                    best_mse_loss=val_mse_loss
                    best_model=self.model
                    torch.save(best_model.state_dict(),f"{path}/model.pt")
                counter+=1

            if counter>=self.patience:
                logger.warning(f"Training stops after {epoch} epochs")
                break

            trial.report(strict_best_total_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        performances={
            "epoch_train_total_losses": torch.tensor(train_epoch_total_losses),
            "epoch_train_contrastive_losses": torch.tensor(train_epoch_contrastive_losses),
            "epoch_train_MSE_losses": torch.tensor(train_epoch_mse_losses),
            "epoch_validation_total_losses":torch.tensor(val_epoch_total_losses),
            "epoch_validation_contrastive_losses":torch.tensor(val_epoch_contrastive_losses),
            "epoch_validation_MSE_losses":torch.tensor(val_epoch_mse_losses)
        }

        torch.save(performances,f"{path}/epochs_performances.pt")

        with mlflow.start_run(run_name=f"lr_{self.lr}_weight_decay_{self.weight_decay}_dropout_{self.model.dropout}"):
            mlflow.log_params({
                "lr":self.lr,
                "weight_decay":self.weight_decay,
                "dropout":self.model.dropout
            })

            mlflow.log_metrics({"best_validation_loss":strict_best_total_loss,
                                "best_contrastive_loss":best_contrastive_loss,
                                "best_MSE_loss":best_mse_loss})

            for epoch,(train_total_loss,train_contrastive_loss,train_mse_loss,val_total_loss,val_contrastive_loss,val_mse_loss) in enumerate(zip(train_epoch_total_losses,train_epoch_contrastive_losses,train_epoch_mse_losses,val_epoch_total_losses,val_epoch_contrastive_losses,val_epoch_mse_losses)):
                mlflow.log_metrics({"epoch_train_total_loss":train_total_loss,
                                    "epoch_train_contrastive_loss":train_contrastive_loss,
                                    "epoch_train_mse_loss":train_mse_loss,
                                    "epoch_validation_total_loss":val_total_loss,
                                    "epoch_validation_contrastive_loss":val_contrastive_loss,
                                    "epoch_validation_mse_loss":val_mse_loss,
                                    },step=epoch)
            
            model_name=f"best_model_lr_{self.lr}_weight_decay_{self.weight_decay}_dropout_{self.model.dropout}"
            model_name=model_name.replace(".",",")
            mlflow.set_tags({"lr": self.lr, "weight_decay": self.weight_decay, "dropout": self.model.dropout})
            mlflow.pytorch.log_model(best_model, model_name)

            return strict_best_total_loss

                    

