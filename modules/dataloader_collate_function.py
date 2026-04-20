from torch.utils.data import DataLoader
import torch


class BertCollateFunction():
    def __init__(self,tokenizer):
        self.tokenizer=tokenizer
    def collate_fn(self,batch):
        full_text=[b["full_text"] for b in batch]
        tokenized_full_text=self.tokenizer(full_text,return_tensors="pt",max_length=512,truncation=True,padding="max_length")
        return tokenized_full_text

class CollateFunction():
        
    def collate_fn(self,
                    batch):
        user_id=[b["user_id"] for b in batch]
        recipe_id=[b["recipe_id"] for b in batch]
        rating_scaled=[b["rating_scaled"] for b in batch]
        i=[b["i"] for b in batch]
        technique_recipes=[torch.tensor(b["techniques_recipes"]) for b in batch]
        calorie_level_scaled=[b["calorie_level_scaled"] for b in batch]
        ingredient_ids=[torch.tensor(b["ingredient_ids"]) for b in batch]
        ingredient_ids_continuous=[torch.tensor(b["ingredient_ids_continuous"]) for b in batch]
        techniques_users=[torch.tensor(b["techniques_users"]) for b in batch]
        items=[torch.tensor(b["items"]) for b in batch]
        n_items_scaled=[b["n_items_scaled"] for b in batch]
        ratings_scaled=[torch.tensor(b["ratings_scaled"]) for b in batch]
        n_ratings_scaled=[b["n_ratings_scaled"] for b in batch]
        n_steps_scaled=[b["n_steps_scaled"] for b in batch]
        minutes_scaled=[b["minutes"] for b in batch]
        nutrition=[torch.tensor(b["nutrition"]) for b in batch]
        n_ingredients_scaled=[b["n_ingredients_scaled"] for b in batch]

        cls_embeddings=[b["cls_embeddings"] for b in batch]
        mean_embeddings=[b["mean_embeddings"] for b in batch]

        inputs={
            "user_id": torch.tensor(user_id,dtype=torch.long).view(-1,1),
            "recipe_id": torch.tensor(recipe_id, dtype=torch.long).view(-1,1),
            "rating_scaled": torch.tensor(rating_scaled, dtype=torch.float32).view(-1,1),
            "i": torch.tensor(i, dtype=torch.long).view(-1,1),
            "technique_recipes": torch.stack(technique_recipes).float(),
            "calorie_level_scaled": torch.tensor(calorie_level_scaled, dtype=torch.float32).view(-1,1),
            "techniques_users": torch.stack(techniques_users).float(),
            "items": torch.stack(items).long(),
            "n_items_scaled": torch.tensor(n_items_scaled, dtype=torch.float32).view(-1,1),
            "ratings_scaled": torch.stack(ratings_scaled).float(),
            "n_ratings_scaled": torch.tensor(n_ratings_scaled, dtype=torch.float32).view(-1,1),
            "minutes_scaled": torch.tensor(minutes_scaled, dtype=torch.float32).view(-1,1),
            "nutrition": torch.stack(nutrition).float(),
            "n_ingredients_scaled": torch.tensor(n_ingredients_scaled, dtype=torch.float32).view(-1,1),
            "n_steps_scaled":torch.tensor(n_steps_scaled,dtype=torch.float32).view(-1,1),
            "ingredient_ids_continuous":torch.stack(ingredient_ids_continuous).long(),
            "cls_embeddings":torch.stack(cls_embeddings,dim=0),
            "mean_embeddings":torch.stack(mean_embeddings,dim=0)

        }

        return inputs
    
class CollateFunctionInferenceRecipes():
        
    def collate_fn(self,
                    batch):
        recipe_id=[b["recipe_id"] for b in batch]
        i=[b["i"] for b in batch]
        techniques_recipes=[torch.tensor(b["techniques_recipes"]) for b in batch]
        calorie_level_scaled=[b["calorie_level_scaled"] for b in batch]
        ingredient_ids=[torch.tensor(b["ingredient_ids"]) for b in batch]
        ingredient_ids_continuous=[torch.tensor(b["ingredient_ids_continuous"]) for b in batch]
        minutes_scaled=[b["minutes"] for b in batch]
        n_steps_scaled=[b["n_steps_scaled"] for b in batch]
        nutrition=[torch.tensor(b["nutrition"]) for b in batch]
        n_ingredients_scaled=[b["n_ingredients_scaled"] for b in batch]
        cls_embeddings=[b["cls_embeddings"] for b in batch]
        mean_embeddings=[b["mean_embeddings"] for b in batch]

        inputs={
            "recipe_id": torch.tensor(recipe_id, dtype=torch.long).view(-1,1),
            "i": torch.tensor(i, dtype=torch.long).view(-1,1),
            "technique_recipes": torch.stack(techniques_recipes).float(),
            "calorie_level_scaled": torch.tensor(calorie_level_scaled, dtype=torch.float32).view(-1,1),
            "ingredient_ids": torch.stack(ingredient_ids).float(),
            "minutes_scaled": torch.tensor(minutes_scaled, dtype=torch.float32).view(-1,1),
            "nutrition": torch.stack(nutrition).float(),
            "n_ingredients_scaled": torch.tensor(n_ingredients_scaled, dtype=torch.float32).view(-1,1),
            "ingredient_ids_continuous":torch.stack(ingredient_ids_continuous).long(),
            "n_steps_scaled":torch.tensor(n_steps_scaled,dtype=torch.float32).view(-1,1),
            "cls_embeddings":torch.stack(cls_embeddings,dim=0),
            "mean_embeddings":torch.stack(mean_embeddings,dim=0)

        }
        return inputs

class CollateFunctionInferenceUsers():
        
    def collate_fn(self,
                    batch):
        user_id=[b["user_id"] for b in batch]
        rating_scaled=[b["rating_scaled"] for b in batch]
        techniques_users=[torch.tensor(b["techniques_users"]) for b in batch]
        items=[torch.tensor(b["items"]) for b in batch]
        n_items_scaled=[b["n_items_scaled"] for b in batch]
        ratings_scaled=[torch.tensor(b["ratings_scaled"]) for b in batch]
        n_ratings_scaled=[b["n_ratings_scaled"] for b in batch]
        inputs={
            "user_id": torch.tensor(user_id,dtype=torch.long).view(-1,1),
            "rating_scaled": torch.tensor(rating_scaled, dtype=torch.float32).view(-1,1),
            "techniques_users": torch.stack(techniques_users).float(),
            "items": torch.stack(items).long(),
            "n_items_scaled": torch.tensor(n_items_scaled, dtype=torch.float32).view(-1,1),
            "ratings_scaled": torch.stack(ratings_scaled).float(),
            "n_ratings_scaled": torch.tensor(n_ratings_scaled, dtype=torch.float32).view(-1,1),
        }

        return inputs
    

    #add n_steps when we'll have the time