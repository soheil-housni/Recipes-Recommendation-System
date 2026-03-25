import torch 
import torch.nn as nn
from torch.nn import MSELoss


class ContrastiveMSELoss(nn.Module):
    def __init__(self,alpha):
        super().__init__()
        self.alpha=alpha
        self.mse_loss=MSELoss()
        
    def forward(self,output,ratings_scaled):
        
        """
        unique_user_ids=[int(x.item()) for x in torch.unique(output["user_id"])]
        unique_user_ids.sort()
        unique_recipe_ids=[int(x.item()) for x in torch.unique(output["recipe_id"])]
        unique_recipe_ids.sort()
        """

        ratings_scaled=ratings_scaled.squeeze(1)

        all_user_embeddings=torch.cat([output["user_id"],output["user_embedding"]],dim=1)
        all_user_embeddings=torch.unique(all_user_embeddings,dim=0)
        
        all_recipe_embeddings=torch.cat([output["recipe_id"],output["recipe_embedding"]],dim=1)
        all_recipe_embeddings=torch.unique(all_recipe_embeddings,dim=0)

        map_user_indices={int(all_user_embeddings[i,0]):i for i in range(len(all_user_embeddings))}
        map_recipe_indices={int(all_recipe_embeddings[i,0]):i for i in range(len(all_recipe_embeddings))}

        u_idx=torch.tensor([map_user_indices[output["user_id"][k].item()] for k in range(len(output["user_id"]))]).long()
        i_idx=torch.tensor([map_recipe_indices[output["recipe_id"][k].item()] for k in range(len(output["recipe_id"]))]).long()

        ratings_matrix=torch.full([len(all_user_embeddings),len(all_recipe_embeddings)],0.1)
        ratings_matrix[u_idx,i_idx]=ratings_scaled

        cos_similarities_matrix=nn.functional.cosine_similarity(all_user_embeddings[:,1:].unsqueeze(1),all_recipe_embeddings[:,1:].unsqueeze(0),dim=2)
        exp_cos_similarities_matrix=torch.exp(cos_similarities_matrix)

        sum_exp_cos=exp_cos_similarities_matrix.sum(dim=1).unsqueeze(1)
        log_probs=torch.log(exp_cos_similarities_matrix/sum_exp_cos)
        contrastive_loss_matrix_users=-(log_probs*ratings_matrix).sum(dim=1)
        contrastive_loss_matrix_users=contrastive_loss_matrix_users.mean()

        sum_exp_cos=exp_cos_similarities_matrix.sum(dim=0).view(-1,1)
        log_probs=torch.log(exp_cos_similarities_matrix/sum_exp_cos)
        contrastive_loss_matrix_recipes=-(log_probs*ratings_matrix).sum(dim=0)
        contrastive_loss_matrix_recipes=contrastive_loss_matrix_recipes.mean()

        contrastive_loss=(contrastive_loss_matrix_users+contrastive_loss_matrix_recipes)/2

        mse_loss=self.mse_loss(ratings_scaled,output["cos_similarities_scaled"])
        total_loss=self.alpha*contrastive_loss+(1-self.alpha)*mse_loss

        return total_loss
