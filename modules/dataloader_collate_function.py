from torch.utils.data import DataLoader
import torch

class CollateFunction():
    def __init__(self,tokenizer):
        self.tokenizer=tokenizer

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
        ratings_scaled=[torch.tensor(b["ratings"]) for b in batch]
        n_ratings_scaled=[b["n_ratings_scaled"] for b in batch]
        minutes_scaled=[b["minutes"] for b in batch]
        nutrition=[torch.tensor(b["nutrition"]) for b in batch]
        n_ingredients_scaled=[b["n_ingredients_scaled"] for b in batch]

        steps=[b["steps"] for b in batch]
        name=[b["name"] for b in batch]
        description=[b["description"] for b in batch]
        tags=[b["tags"] for b in batch]
        full_text=[b["full_text"] for b in batch]


        tokenized_steps=self.tokenizer(steps,return_tensors="pt",max_length=256,truncation=True,padding=True)
        tokenized_names=self.tokenizer(name,return_tensors="pt",max_length=128,truncation=True,padding=True)
        tokenized_descriptions=self.tokenizer(description,return_tensors="pt",max_length=256,truncation=True,padding=True)
        tokenized_tags=self.tokenizer(tags,return_tensors="pt",max_length=256,truncation=True,padding=True)

        tokenized_steps={key+"_steps":tokenized_steps[key] for key in list(tokenized_steps.keys())}
        
        tokenized_names={key+"_names":tokenized_names[key] for key in list(tokenized_names.keys())}

        tokenized_descriptions={key+"_descriptions":tokenized_descriptions[key] for key in list(tokenized_descriptions.keys())}

        tokenized_tags={key+"_tags":tokenized_tags[key] for key in list(tokenized_tags.keys())}

        tokenized_full_text=self.tokenizer(full_text,return_tensors="pt",max_length=512,truncation=True,padding=True)
        tokenized_full_text={key+"_full":tokenized_tags[key] for key in list(tokenized_tags.keys())}



        inputs={
            "user_id": torch.tensor(user_id).view(-1,1),
            "recipe_id": torch.tensor(recipe_id).view(-1,1),
            "rating_scaled": torch.tensor(rating_scaled).view(-1,1),
            "i": torch.tensor(i).view(-1,1),
            "technique_recipes": torch.stack(technique_recipes),
            "calorie_level_scaled": torch.tensor(calorie_level_scaled).view(-1,1),
            "ingredient_ids": torch.stack(ingredient_ids),
            "techniques_users": torch.stack(techniques_users),
            "items": torch.stack(items),
            "n_items_scaled": torch.tensor(n_items_scaled).view(-1,1),
            "ratings_scaled": torch.stack(ratings_scaled),
            "n_ratings_scaled": torch.tensor(n_ratings_scaled).view(-1,1),
            "minutes_scaled": torch.tensor(minutes_scaled).view(-1,1),
            "nutrition": torch.stack(nutrition),
            "n_ingredients_scaled": torch.tensor(n_ingredients_scaled).view(-1,1),
            "ingredient_ids_continuous":torch.stack(ingredient_ids_continuous),
        }

        list_tokenized_dicts=[tokenized_steps,tokenized_names,tokenized_descriptions,tokenized_tags,tokenized_full_text]

        for dict in list_tokenized_dicts:
            inputs.update(dict)

        return inputs

