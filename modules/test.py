import torch
from .loss_function import ContrastiveMSELoss
import numpy as np
from loguru import logger
import mlflow

class Test():
    def __init__(self,
                 test_dataloader,
                 model,
                 device,
                 loss_alpha:float=0.5,
                 loss_temperature:float=0.2,):
        
        self.test_dataloader=test_dataloader
        self.model=model
        self.device=device
        self.loss_alpha=loss_alpha
        self.loss_temparature=loss_temperature

        self.model=self.model.to(self.device)
        self.criterion=ContrastiveMSELoss(alpha=self.loss_alpha,temperature=self.loss_temparature)
    
    def run_testing(self,path,run_id):
        logger.info("Start of the test :")
        self.model.eval()
        test_batch_total_losses=[]
        test_batch_contrastive_losses=[]
        test_batch_mse_losses=[]
        with torch.inference_mode():
            for batch in self.test_dataloader:
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
                                   n_steps_scaled=batch["n_steps_scaled"],
                                   cls_embeddings=batch["cls_embeddings"],
                                   mean_embeddings=batch["mean_embeddings"]
                                   )
                
                loss=self.criterion(
                    outputs=outputs,
                    rating_scaled=batch["rating_scaled"],
                )

                test_batch_total_losses.append(loss.item())
                test_batch_contrastive_losses.append(self.criterion.access_loss_components()["Contrastive loss"])
                test_batch_mse_losses.append(self.criterion.access_loss_components()["MSE loss"])
            
            test_total_loss=np.mean(test_batch_total_losses)
            test_contrastive_loss=np.mean(test_batch_contrastive_losses)
            test_mse_loss=np.mean(test_batch_mse_losses)

            logger.info(f"Test Total loss = {test_total_loss}")
            logger.info(f"Test Contrastive loss = {test_contrastive_loss}")
            logger.info(f"Test MSE loss = {test_mse_loss}")

            metrics={
                "Test Total loss":test_total_loss,
                "Test Contrastive loss":test_contrastive_loss,
                "Test MSE loss":test_mse_loss
            }

            with mlflow.start_run(run_id=run_id):
                mlflow.log_metrics(metrics)
            torch.save(metrics,f"{path}/test_metrics.pt")

        return test_total_loss,test_contrastive_loss,test_mse_loss
