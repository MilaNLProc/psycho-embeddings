from transformers import AutoTokenizer, AutoModel


class BaseHFEmbedder:

    def __init__(self):
        self.model = None
        self.tokenizer = None

    def get_single_embedding(self, sentence: str, word: str, layer: int):
        tok = self.tokenizer([sentence], return_tensors="pt")
        idx = self.tokenizer.convert_ids_to_tokens(list(tok["input_ids"][0])).index(word)

        # index 0 is the first sentence (there's only one sentence in the list)
        return self.model(**tok)["hidden_states"][layer][0][idx].detach().numpy()


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
