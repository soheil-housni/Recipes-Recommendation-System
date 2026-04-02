import torch 
import torch.nn as nn
from torch.nn import MSELoss


class ContrastiveMSELoss(nn.Module):
    def __init__(self,alpha:int=0.5):
        super().__init__()
        self.alpha=alpha
        self.mse_loss=MSELoss()

    def access_loss_components(self):
        return {"Contrastive loss":self.contrastive_loss.item(),"MSE loss":self.mse_loss_value.item(),"Total loss":self.total_loss.item()}
        
    def forward(self,outputs,rating_scaled):
        
        """
        unique_user_ids=[int(x.item()) for x in torch.unique(output["user_id"])]
        unique_user_ids.sort()
        unique_recipe_ids=[int(x.item()) for x in torch.unique(output["recipe_id"])]
        unique_recipe_ids.sort()
        """
        #corriger mask fill
        
        user_embeddings=outputs["user_embeddings"]
        recipe_embeddings=outputs["recipe_embeddings"]
        N=len(user_embeddings)
        cos_similarities_matrix=nn.functional.cosine_similarity(user_embeddings.unsqueeze(0),recipe_embeddings.unsqueeze(1),dim=2)
        logits=nn.functional.log_softmax(cos_similarities_matrix,dim=1)
        ratings_scaled_matrix=torch.full((N,N),0.1,device=user_embeddings.device)
        ratings_scaled_matrix[torch.arange(N),torch.arange(N)]=rating_scaled.squeeze()
        contrastive_loss=logits*ratings_scaled_matrix
        self.contrastive_loss=-(contrastive_loss.sum(dim=1)).mean()
        rating_scaled=rating_scaled.squeeze(1)
        self.mse_loss_value=self.mse_loss(rating_scaled,outputs["cos_similarities_scaled"])
        self.total_loss=self.alpha*self.contrastive_loss+(1-self.alpha)*self.mse_loss_value
        return self.total_loss




        """
        rating_scaled=rating_scaled.squeeze(1)

        user_ids=user_ids.detach().long()
        all_user_embeddings=outputs["user_embeddings"]
        user_ids=user_ids.squeeze(1)
        unique_user_ids,inverse_indices=torch.unique(user_ids,dim=0,sorted=True,return_inverse=True)
        unique_user_indices=torch.empty(len(unique_user_ids),dtype=torch.long)
        original_indicies=torch.arange(len(all_user_embeddings)).long()
        unique_user_indices.scatter_(dim=0,src=original_indicies,index=inverse_indices)
        all_user_embeddings=all_user_embeddings[unique_user_indices]
        filtered_user_ids=user_ids[unique_user_indices]
        
        recipe_ids=recipe_ids.detach().long()
        all_recipe_embeddings=outputs["recipe_embeddings"]
        recipe_ids=recipe_ids.squeeze(1)
        unique_recipe_ids,inverse_indices=torch.unique(recipe_ids,dim=0,return_inverse=True,sorted=True)
        original_indicies=torch.arange(len(all_recipe_embeddings)).long()
        unique_recipe_indices=torch.empty(len(unique_recipe_ids),dtype=torch.long)
        unique_recipe_indices.scatter_(dim=0,index=inverse_indices,src=original_indicies)
        all_recipe_embeddings=all_recipe_embeddings[unique_recipe_indices]
        filtered_recipe_ids=recipe_ids[unique_recipe_indices]

        map_user_indices={int(filtered_user_ids[i].item()):i for i in range(len(filtered_user_ids))}
        map_recipe_indices={int(filtered_recipe_ids[i].item()):i for i in range(len(filtered_recipe_ids))}


        u_idx=torch.tensor([map_user_indices[int(user_ids[k].item())] for k in range(len(user_ids))]).long()
        i_idx=torch.tensor([map_recipe_indices[int(recipe_ids[k].item())] for k in range(len(recipe_ids))]).long()

        ratings_matrix=torch.full([len(all_user_embeddings),len(all_recipe_embeddings)],0.1,device=rating_scaled.device).index_put((u_idx,i_idx),rating_scaled)

        cos_similarities_matrix=nn.functional.cosine_similarity(all_user_embeddings.unsqueeze(1),all_recipe_embeddings.unsqueeze(0),dim=2)
        #exp_cos_similarities_matrix=torch.exp(cos_similarities_matrix)

        #sum_exp_cos=exp_cos_similarities_matrix.sum(dim=1).unsqueeze(1)
        #log_probs=torch.log(exp_cos_similarities_matrix/sum_exp_cos)
        log_probs_users=nn.functional.log_softmax(cos_similarities_matrix,dim=1)
        contrastive_loss_matrix_users=-(log_probs_users*ratings_matrix).sum(dim=1)
        contrastive_loss_matrix_users=contrastive_loss_matrix_users.mean()

        #sum_exp_cos=exp_cos_similarities_matrix.sum(dim=1).view(-1,1)
        #log_probs=torch.log(exp_cos_similarities_matrix/sum_exp_cos)
        log_probs_recipes=nn.functional.log_softmax(torch.transpose(cos_similarities_matrix,dim0=0,dim1=1),dim=1)
        contrastive_loss_matrix_recipes=-(log_probs_recipes*torch.transpose(ratings_matrix,dim0=0,dim1=1)).sum(dim=1)
        contrastive_loss_matrix_recipes=contrastive_loss_matrix_recipes.mean()

        contrastive_loss=(contrastive_loss_matrix_users+contrastive_loss_matrix_recipes)/2

        mse_loss=self.mse_loss(rating_scaled,outputs["cos_similarities_scaled"])
        total_loss=self.alpha*contrastive_loss+(1-self.alpha)*mse_loss

        return total_loss
        """