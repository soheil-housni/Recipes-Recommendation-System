import torch
import mmh3
import torch.nn
import numpy as np

class EncodedHashedEmbeddings():
    def __init__(self,
                 n_recipes_ids,
                 recipe_id_emb_dim,
                 n_ingredients_ids,
                 ingredient_id_emb_dim
                 ):
        super().__init__()

        self.n_recipe_ids=n_recipes_ids
        self.recipes_bucket_size=int(0.2*self.n_recipe_ids)
        self.recipe_id_emb_dim=recipe_id_emb_dim

        self.n_ingredients_ids=n_ingredients_ids
        self.ingredients_bucket_size=int(0.2*self.n_ingredients_ids)
        self.ingredient_id_emb_dim=ingredient_id_emb_dim
    
    def get_encoded_hashed_embeddings(self,
                              path:str):
        

        hash_func=np.vectorize(lambda i,k: mmh3.hash(str(int(i)), seed=int(k)))
        I,K=np.meshgrid(np.arange(self.n_recipe_ids+2),np.arange(self.recipe_id_emb_dim),indexing="ij")
        HASH=hash_func(I,K)% self.recipes_bucket_size
        recipes_ids_encoded_embeddings= torch.tensor(2*(HASH/(self.recipes_bucket_size-1))-1,dtype=torch.float32)
        torch.save(recipes_ids_encoded_embeddings,f"{path}/hashed_embeddings_recipes.pt")



        hash_func=np.vectorize(lambda i,k: mmh3.hash(str(int(i)), seed=int(k)))
        I,K=np.meshgrid(np.arange(self.n_ingredients_ids+2),np.arange(self.ingredient_id_emb_dim),indexing="ij")
        HASH=hash_func(I,K)% self.ingredients_bucket_size
        ingredients_ids_encoded_embeddings= torch.tensor(2*(HASH/(self.ingredients_bucket_size-1))-1,dtype=torch.float32)
        torch.save(ingredients_ids_encoded_embeddings,f"{path}/hashed_embeddings_ingredients.pt")

        return recipes_ids_encoded_embeddings,ingredients_ids_encoded_embeddings






        """
        recipes_ids_encoded_embeddings=torch.zeros((self.n_recipe_ids+1,self.recipe_id_emb_dim))
        for i in range(self.n_recipe_ids+1):
            h = torch.tensor([mmh3.hash(str(i), seed=k) % self.recipes_bucket_size for k in range(self.recipe_id_emb_dim)])
            recipes_ids_encoded_embeddings[i,:] = 2*(h/self.recipes_bucket_size)-1
        torch.save(recipes_ids_encoded_embeddings,f"{path}/hashed_embeddings_recipes.pt")

        ingredients_ids_encoded_embeddings=torch.zeros((self.n_ingredients_ids+1,self.ingredient_id_emb_dim))
        for i in range(self.n_ingredients_ids+1):
            h = torch.tensor([mmh3.hash(str(i), seed=k) % self.ingredients_bucket_size for k in range(self.ingredient_id_emb_dim)])
            ingredients_ids_encoded_embeddings[i,:]=2*(h/self.ingredients_bucket_size)-1
        torch.save(ingredients_ids_encoded_embeddings,f"{path}/hashed_embeddings_ingredients.pt")
        
        return recipes_ids_encoded_embeddings,ingredients_ids_encoded_embeddings
        """
