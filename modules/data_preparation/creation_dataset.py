from torch.utils.data import Dataset


class BERTCreationDataset(Dataset):
    def __init__(self,
                 df_text):
        self.df_text=df_text
    def __len__(self):
        return len(self.df_text)
    def __getitem__(self,
                    idx):
        item={
            "full_text":self.df_text.iloc[idx]
        }
        return item


class CreationDataset(Dataset):
    def __init__(self,
                 df,
                 cls_embeddings,
                 mean_embeddings):
        self.df=df
        self.cls_embeddings=cls_embeddings
        self.mean_embeddings=mean_embeddings
    
    def __len__(self):
        if (len(self.df)!=len(self.cls_embeddings)) or (len(self.df)!=len(self.mean_embeddings)) or (len(self.cls_embeddings)!=len(self.mean_embeddings)):
            raise ValueError("The three components should be of the same size")

        return len(self.df)

    def __getitem__(self,
                    idx):
        item={
            col:self.df.iloc[idx][col] for col in self.df.columns
        }
        item.update({"cls_embeddings":self.cls_embeddings[idx],"mean_embeddings":self.mean_embeddings[idx]})
        return item