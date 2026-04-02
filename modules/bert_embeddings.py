import torch

class BERTEmbeddingsExtractor():
    def __init__(self,
                 bert_model,
                 device):
        self.bert_model=bert_model
        self.device=device
        self.bert_model.to(self.device)

    def get_bert_embeddings(self,dataloader,path):
        all_mean_embeddings=[]
        all_cls_embeddings=[]

        self.bert_model.eval()
        with torch.inference_mode():
            for batch in dataloader:
                for key in list(batch.keys()):
                    batch[key]=batch[key].to(self.device)
                output=self.bert_model(input_ids=batch["input_ids"],attention_mask=batch["attention_mask"]).last_hidden_state
                cls_embeddings=output[:,0]
                mean_embeddings=output.mean(dim=1)
                all_cls_embeddings.append(cls_embeddings)
                all_mean_embeddings.append(mean_embeddings)
        
        all_cls_embeddings=torch.cat(all_cls_embeddings,dim=0)
        all_mean_embeddings=torch.cat(all_mean_embeddings,dim=0)

        torch.save(all_cls_embeddings,f"{path}/all_cls_embeddings.pt")
        torch.save(all_mean_embeddings,f"{path}/all_mean_embeddings.pt")
