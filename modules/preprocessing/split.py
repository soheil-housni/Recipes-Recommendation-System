from sklearn.model_selection import train_test_split
import torch


def split_df(df,
             cls_embeddings,
             mean_embeddings,
             random_state:int=42):
    cls_embeddings=cls_embeddings.detach().clone().numpy()
    mean_embeddings=mean_embeddings.detach().clone().numpy()
    temp_df,test_df,temp_cls_embeddings,test_cls_embeddings,temp_mean_embeddings,test_mean_embeddings=train_test_split(df,cls_embeddings,mean_embeddings,test_size=0.15,random_state=random_state)
    train_df,val_df,train_cls_embeddings,val_cls_embeddings,train_mean_embeddings,val_mean_embeddings=train_test_split(temp_df,temp_cls_embeddings,temp_mean_embeddings,test_size=0.15/0.85,random_state=random_state)

    train_cls_embeddings=torch.tensor(train_cls_embeddings)
    val_cls_embeddings=torch.tensor(val_cls_embeddings)
    test_cls_embeddings=torch.tensor(test_cls_embeddings)

    train_mean_embeddings=torch.tensor(train_mean_embeddings)
    val_mean_embeddings=torch.tensor(val_mean_embeddings)
    test_mean_embeddings=torch.tensor(test_mean_embeddings)
    

    return train_df,val_df,test_df,train_cls_embeddings,val_cls_embeddings,test_cls_embeddings,train_mean_embeddings,val_mean_embeddings,test_mean_embeddings