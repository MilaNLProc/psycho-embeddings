from transformers import AutoTokenizer, AutoModel
import torch
from typing import List
from datasets import Dataset
from tqdm import tqdm
import numpy as np
from psycho_embeddings.feature_extractor import NewFeatureExtractionPipeline
import datasets


def find_sub_list(sl, l):
    results = list()
    sll = len(sl)
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if l[ind:ind + sll] == sl:
            results.append((ind, ind + sll))

    return results



class GPT2Embedder:

    def __init__(self, layer:int, device:int = 0):
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.model = AutoModel.from_pretrained("gpt2", output_hidden_states=True).to("cuda")
        self.device = device


        self.feature = NewFeatureExtractionPipeline(layer=layer, model=self.model, tokenizer=self.tokenizer, device=device)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})


    def get_single_embedding(self, word: str):

        embedding_middle = self.feature([f" {word}"])
        embedding_start = self.feature([f"{word}"])

        embedding_middle = np.mean(embedding_middle[0][0], axis=0)
        embedding_start = np.mean(embedding_start[0][0], axis=0)

        return embedding_start, embedding_middle

    def get_embedding_from_dataset(self, words: List[str], texts: List[str], **kwargs):
        max_seq_length = kwargs.get("max_seq_length", 500)

        dataset = self._tokenize_dataset(texts, max_seq_length)
        dataset.set_format("pt")

        features_from_model = self.feature(texts)

        which_tokenization = []
        for word, t in zip(words, texts):

            if t.split()[0] == word:
                tok_word_start = self.tokenizer(word, add_special_tokens=False)
                which_tokenization.append(tok_word_start)
            else:
                tok_word_middle = self.tokenizer(f" {word}", add_special_tokens=False)
                which_tokenization.append(tok_word_middle)

        idx = [
            find_sub_list(tok_word["input_ids"], input_ids.tolist())[0]
            for tok_word, input_ids in zip(which_tokenization, dataset["input_ids"])
        ]

        # average over sub tokens
        embeddings = list()

        for hs, (l_idx, r_idx) in zip(features_from_model, idx):
            word_embeddings = hs[0][l_idx:r_idx]
            if len(word_embeddings) > 1:
                word_embeddings = np.mean(word_embeddings, axis=0)
            embeddings.append(word_embeddings)

        # average over the dataset
        #embedding = np.mean(embeddings, axis=0)

        return embeddings

    def _tokenize_dataset(self, texts, max_seq_length):
        d = {"text": texts}
        dataset = Dataset.from_dict(d)

        def tokenize_text(examples):
            return self.tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_seq_length)

        datasets.set_progress_bar_enabled(False)
        # tokenize the corpus
        dataset = dataset.map(tokenize_text, batched=True, desc="Tokenizing", remove_columns=["text"])
        return dataset


class BERTEmbedder:

    def __init__(self, layer: int, size:str ="base", device:int = 0):
        super().__init__()
        self.device = device
        if size == "base":
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.model = AutoModel.from_pretrained("bert-base-uncased", output_hidden_states=True).to("cuda")
        elif size == "large":
            self.tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
            self.model = AutoModel.from_pretrained("bert-large-uncased", output_hidden_states=True).to("cuda")
        else:
            raise NotImplemented()

        self.feature = NewFeatureExtractionPipeline(layer=layer, model=self.model, tokenizer=self.tokenizer, device=self.device)

    def get_single_embedding(self, word: str):

        embedding_start = self.feature([f"{word}"])

        embedding_start = np.mean(embedding_start[0][0], axis=0)

        return embedding_start

    def get_embedding_from_dataset(self, words: List[str], texts: List[str], **kwargs):
        max_seq_length = kwargs.get("max_seq_length", 500)

        dataset = self._tokenize_dataset(texts, max_seq_length)
        dataset.set_format("pt")

        features_from_model = self.feature(texts)

        which_tokenization = []
        for word in words:
            tok_word_start = self.tokenizer(word, add_special_tokens=False)
            which_tokenization.append(tok_word_start)

        idx = [
            find_sub_list(tok_word["input_ids"], input_ids.tolist())[0]
            for tok_word, input_ids in zip(which_tokenization, dataset["input_ids"])
        ]

        # average over sub tokens
        embeddings = list()

        for hs, (l_idx, r_idx) in zip(features_from_model, idx):
            word_embeddings = hs[0][l_idx:r_idx]
            if len(word_embeddings) > 1:
                word_embeddings = np.mean(word_embeddings, axis=0)
            embeddings.append(word_embeddings)

        # average over the dataset
        embedding = np.mean(embeddings, axis=0)

        return embedding

    def _tokenize_dataset(self, texts, max_seq_length):
        d = {"text": texts}
        dataset = Dataset.from_dict(d)

        def tokenize_text(examples):
            return self.tokenizer(examples["text"],
                                  padding="max_length",
                                  truncation=True, max_length=max_seq_length)

        datasets.set_progress_bar_enabled(False)
        # tokenize the corpus
        dataset = dataset.map(tokenize_text, batched=True, desc="Tokenizing", remove_columns=["text"])
        return dataset

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
        max_seq_length = kwargs.get("max_seq_length", 200)

        dataset = self._tokenize_dataset(texts, max_seq_length)
        dataset.set_format("pt")

        feature = NewFeatureExtractionPipeline(layer=layer, model=self.model, tokenizer=self.tokenizer)

        features_from_model = feature(texts)

        tok_word_middle = self.tokenizer(f" {word}", add_special_tokens=False)
        tok_word_start = self.tokenizer(word, add_special_tokens=False)

        which_tokenization = []
        for t in texts:
            if t.split()[0] == word:
                which_tokenization.append(tok_word_start)
            else:
                which_tokenization.append(tok_word_middle)

        idx = [
            find_sub_list(tok_word["input_ids"], input_ids.tolist())[0]
            for tok_word, input_ids in zip(which_tokenization, dataset["input_ids"])
        ]

        # average over sub tokens
        embeddings = list()

        for hs, (l_idx, r_idx) in zip(features_from_model, idx):
            word_embeddings = hs[0][l_idx:r_idx]
            if len(word_embeddings) > 1:
                word_embeddings = np.mean(word_embeddings, axis=0)
            embeddings.append(word_embeddings)

        # average over the dataset
        embedding = np.mean(embeddings, axis=0)

        return embedding


    def _tokenize_dataset(self, texts, max_seq_length):
        d = {"text": texts}
        dataset = Dataset.from_dict(d)

        def tokenize_text(examples):
            return self.tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_seq_length)

        # tokenize the corpus
        dataset = dataset.map(tokenize_text, batched=True, desc="Tokenizing", remove_columns=["text"])
        return dataset
