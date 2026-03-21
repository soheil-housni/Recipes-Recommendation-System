from torch.utils.data import DataLoader
import torch

class CollateFunction():
    def __init__(self,tokenizer):
        self.tokenizer=tokenizer

    def CollateFunction(self,
                        batch):
        user_id=[b["user_id"] for b in batch]
        recipe_id=[b["recipe_id"] for b in batch]
        rating=[b["rating"] for b in batch]
        i=[b["i"] for b in batch]
        technique_recipes=[torch.tensor(b["techniques_recipes"]) for b in batch]
        calorie_level=[b["calorie_level"] for b in batch]
        ingredient_ids=[torch.tensor(b["ingredient_ids"]) for b in batch]
        techniques_users=[torch.tensor(b["techniques_users"]) for b in batch]
        items=[torch.tensor(b["items"]) for b in batch]
        n_items=[b["n_items"] for b in batch]
        ratings=[torch.tensor(b["ratings"]) for b in batch]
        n_ratings=[b["n_ratings"] for b in batch]
        minutes=[b["minutes"] for b in batch]
        nutrition=[torch.tensor(b["nutrition"]) for b in batch]
        n_ingredients=[b["n_ingredients"] for b in batch]

        steps=[b["steps"] for b in batch]
        name=[b["name"] for b in batch]
        description=[b["description"] for b in batch]
        tags=[b["tags"] for b in batch]


        tokenized_steps=self.tokenizer(steps)
        tokenized_names=self.tokenizer(name)
        tokenized_descriptions=self.tokenizer(description)
        tokenized_tags=self.tokenizer(tags)

        for key in list(tokenized_steps.keys()):
            tokenized_steps[key+"_steps"]=tokenized_steps.pop(key)
        
        for key in list(tokenized_names.keys()):
            tokenized_steps[key+"_names"]=tokenized_steps.pop(key)

        for key in list(tokenized_descriptions.keys()):
            tokenized_steps[key+"_descriptions"]=tokenized_steps.pop(key)

        for key in list(tokenized_tags.keys()):
            tokenized_steps[key+"_tags"]=tokenized_steps.pop(key)

        inputs={
            "user_id": torch.tensor(user_id).view(-1,1),
            "recipe_id": torch.tensor(recipe_id).view(-1,1),
            "rating": torch.tensor(rating).view(-1,1),
            "i": torch.tensor(i).view(-1,1),
            "technique_recipes": torch.stack(technique_recipes),
            "calorie_level": torch.tensor(calorie_level).view(-1,1),
            "ingredient_ids": torch.stack(ingredient_ids),
            "techniques_users": torch.stack(techniques_users),
            "items": torch.stack(items),
            "n_items": torch.tensor(n_items).view(-1,1),
            "ratings": torch.stack(ratings),
            "n_ratings": torch.tensor(n_ratings).view(-1,1),
            "minutes": torch.tensor(minutes).view(-1,1),
            "nutrition": torch.stack(nutrition),
            "n_ingredients": torch.tensor(n_ingredients).view(-1,1),
        }

        list_tokenized_dicts=[tokenized_steps,tokenized_names,tokenized_descriptions,tokenized_tags]

        for dict in list_tokenized_dicts:
            inputs.update(dict)

        return inputs

