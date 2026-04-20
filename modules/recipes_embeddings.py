import torch



class RecipesEmbeddingsExtractor():
    def __init__(self,
                 model,
                 device):
        
        self.model=model
        self.device=device
        self.model.to(self.device)

    def get_recipes_embeddings(self,dataloader,path):
        self.model.eval()
        all_recipes_embeddings=[]
        with torch.inference_mode():
            for batch in dataloader:
                for key in list(batch.keys()):
                    batch[key]=batch[key].to(self.device)
                recipes_embeddings=self.model.forward_recipes(technique_recipes=batch["technique_recipes"],
                                                              calorie_level_scaled=batch["calorie_level_scaled"],
                                                              ingredient_ids_continuous=batch["ingredient_ids_continuous"],
                                                              minutes_scaled=batch["minutes_scaled"],
                                                              nutrition=batch["nutrition"],
                                                              n_ingredients_scaled=batch["n_ingredients_scaled"],
                                                              n_steps_scaled=batch["n_steps_scaled"],
                                                              cls_embeddings=batch["cls_embeddings"],
                                                              mean_embeddings=batch["mean_embeddings"])
                all_recipes_embeddings.append(recipes_embeddings)
        all_recipes_embeddings=torch.cat(all_recipes_embeddings,dim=0)
        torch.save(all_recipes_embeddings,f"{path}/recipes_embeddings.pt")
        return all_recipes_embeddings
