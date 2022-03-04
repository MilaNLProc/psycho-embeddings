from transformers import AutoTokenizer, AutoModel
import torch


def find_sub_list(sl, l):
    results = list()
    sll = len(sl)
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if l[ind:ind + sll] == sl:
            results.append((ind, ind + sll-1))

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


class GPT2Embedder(BaseHFEmbedder):

    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.model = AutoModel.from_pretrained("gpt2", output_hidden_states=True)


class BERTEmbedder(BaseHFEmbedder):

    def __init__(self, size="base"):
        super().__init__()

        if size == "base":
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.model = AutoModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
        else:
            raise NotImplemented()
