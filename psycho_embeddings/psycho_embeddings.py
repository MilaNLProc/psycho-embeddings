from transformers import AutoTokenizer, AutoModel
import torch
from typing import List
from datasets import Dataset
from tqdm import tqdm

    
def find_sub_list(sl, l):
    results = list()
    sll = len(sl)
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if l[ind:ind + sll] == sl:
            results.append((ind, ind + sll))

    return results    


class BaseHFEmbedder:

    def __init__(self):
        self.model = None
        self.tokenizer = None

    def get_single_embedding(self, sentence: str, word: str, layer: int):
        tok_sentence = self.tokenizer([sentence], return_tensors="pt")
        tok_word = self.tokenizer(word, add_special_tokens=False)
        
        idx = find_sub_list(tok_word["input_ids"], tok_sentence["input_ids"][0].tolist())
        idx = idx[0] # consider only the first occurrence 
        
        with torch.no_grad():
            outputs = self.model(**tok_sentence)
        
        # index 0 is the first sentence (there's only one sentence in the list)
        target_emb = outputs["hidden_states"][layer][0][idx[0]:idx[1] + 1]
        # average across subtokens
        target_emb = target_emb.mean(0) 
        
        return target_emb.numpy()
    
        
    def get_embedding_from_dataset(self, word: str, texts: List[str], layer: int, **kwargs):
        batch_size = kwargs.get("batch_size", 16)
        max_seq_length = kwargs.get("max_seq_length", 32)
        
        dataset = self._tokenize_dataset(texts, max_seq_length)
        dataset.set_format("pt")
        
        # inference on the corpus
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        hidden_states = list()
        for batch in tqdm(loader, total=len(loader), desc="Batch"):
            
            breakpoint()
            with torch.no_grad():
                hs = self.model(**batch).hidden_states
                hs = torch.stack(hs) # layers+1 b s hidden
                hs = hs.transpose(0, 1) 
                hidden_states.append(hs)
        
        hidden_states = torch.cat(hidden_states) # dataset_size layer+1 s hidden
        hidden_states = [hs[layer] for hs in hidden_states]
        
        tok_word = self.tokenizer(word, add_special_tokens=False)
        idx = [
            find_sub_list(tok_word["input_ids"], input_ids.tolist())[0]
            for input_ids in dataset["input_ids"]
        ]
        
        assert len(hidden_states) == len(idx)
        
        # average over sub tokens
        embeddings = list()
        for hs, (l_idx, r_idx) in zip(hidden_states, idx):
            embeddings.append(torch.mean(hs[l_idx:r_idx, :], dim=0))
        embeddings = torch.stack(embeddings)
        
        # average over the dataset
        embedding = embeddings.mean(0)
        
        return embedding.numpy()


    def _tokenize_dataset(self, texts, max_seq_length):
        d = {"text": texts}
        dataset = Dataset.from_dict(d)
        
        def tokenize_text(examples):
            return self.tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_seq_length)
        
        # tokenize the corpus
        dataset = dataset.map(tokenize_text, batched=True, desc="Tokenizing", remove_columns=["text"])
        return dataset
        
        


class GPT2Embedder(BaseHFEmbedder):

    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.model = AutoModel.from_pretrained("gpt2", output_hidden_states=True)
        
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})


class BERTEmbedder(BaseHFEmbedder):

    def __init__(self, size="base"):
        super().__init__()

        if size == "base":
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.model = AutoModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
        else:
            raise NotImplemented()
