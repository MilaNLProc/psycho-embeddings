from transformers import (
    AutoTokenizer,
    AutoModel,
    BatchEncoding,
    FeatureExtractionPipeline,
)
import torch
from typing import List, Tuple
from datasets import Dataset
import numpy as np
from psycho_embeddings.feature_extractor import NewFeatureExtractionPipeline
import datasets
from abc import ABC, abstractmethod


def crop_string(text, word, x):
    text = text.lower()
    word = word.lower()

    splitted = text.split()

    return " ".join(splitted[0 : splitted.index(word) + 1])


def find_sub_list(sl, l):
    results = list()
    sll = len(sl)
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if l[ind : ind + sll] == sl:
            results.append((ind, ind + sll))

    return results


class BaseHFEmbedder(ABC):
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.feature = None

    @abstractmethod
    def get_embedding(self, sentence: str, word: str, layer: int):
        pass

    @abstractmethod
    def get_embedding_from_dataset(
        self, word: str, texts: List[str], layer: int, **kwargs
    ):
        pass

    def get_tokens(self, words: List[str], texts: List[str]):
        tokenized_words = self._tokenize_words(words)
        tokenized_texts = self._tokenize_texts(texts)
        return tokenized_words, tokenized_texts

    def embed(self, texts) -> List[Tuple[torch.Tensor]]:
        """
        Extract embeddings with model.

        Returns:
            list of tuples, one per text. Each tuple has (1 + num_layers) elements
        """
        assert self.feature is not None

        features_from_model = self.feature(texts)
        return features_from_model

    def _tokenize_words(self, words: List[str]):
        assert self.tokenizer is not None

        tokenized_words = [self.tokenizer(w, add_special_tokens=False) for w in words]
        return tokenized_words

    def _tokenize_texts(self, texts, max_seq_length=512) -> datasets.Dataset:
        d = {"text": texts}
        dataset = Dataset.from_dict(d)

        def tokenize_text(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=max_seq_length,
            )

        try:
            datasets.set_progress_bar_enabled(False)
        except:
            datasets.logging.disable_progress_bar()

        dataset = dataset.map(
            tokenize_text, batched=True, desc="Tokenizing", remove_columns=["text"]
        )
        dataset.set_format("pt")
        return dataset


class BERTEmbedder(BaseHFEmbedder):
    def __init__(self, size: str = "base", device: int = 0):
        super().__init__()
        self.device = device

        if size == "base":
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.model = AutoModel.from_pretrained(
                "bert-base-uncased", output_hidden_states=True
            )
        elif size == "large":
            self.tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
            self.model = AutoModel.from_pretrained(
                "bert-large-uncased", output_hidden_states=True
            )
        else:
            raise NotImplemented()

        self.feature = NewFeatureExtractionPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            framework="pt",
        )

    def get_embedding(self, word: str, context: str = None, layer: int = None):
        """
        Get a single word embedding. If `context` is defined, the embedding is contextual.
        The first row corresponds to the word embedding layer in BERT.

        Params:
            word: str
            context: str. Optional
            layer: int. Optional, use to extract single layer embedding

        Return:
            torch.tensor. Shape: (num_layers + 1, hidden size) if layer is None, (hidden_size) otherwise
        """
        if len(word.split(" ")) > 1:
            raise ValueError("Use this method to encode single words.")
        if layer < 0 or layer > 12:
            raise ValueError("Layer index not within [0, 12]")
        if word and context and word not in context:
            raise ValueError("Word must be in the context.")

        if context is None:  # single embedding
            embedding_start = self.feature(word)
            embedding_start = torch.cat(embedding_start)

            # mean across subtokens
            embedding_start = embedding_start[:, 1:-1, :].mean(1)

            if layer is not None and layer >= 0:
                embedding_start = embedding_start[layer, :]

            return embedding_start

        else:  # contextualized embedding
            if not isinstance(word, list):
                word = [word]
            if not isinstance(context, list):
                context = [context]

            tokenized_word, tokenized_context = self.get_tokens(word, context)
            features = self.embed([context])

            if layer is not None and layer >= 0:
                features = [f[layer] for f in features]

            embedding = self.get_embedding_from_dataset(
                tokenized_word, tokenized_context, features
            )

            if embedding.shape[0] == 1:
                embedding = embedding.squeeze(0)

            return embedding

    def get_embedding_from_dataset(
        self,
        tokenized_words: List[BatchEncoding],
        tokenized_texts: List[Dataset],
        features,
    ):
        """
        Find the embedding of word_i in texts_i, averaging the embeddings across subtokens.
        """
        assert len(tokenized_words) == len(tokenized_texts)

        idx = [
            find_sub_list(tok_word["input_ids"], input_ids.tolist())[0]
            for tok_word, input_ids in zip(
                tokenized_words, tokenized_texts["input_ids"]
            )
        ]

        # average over sub tokens
        embeddings = list()
        for hs, (l_idx, r_idx) in zip(features, idx):
            word_embeddings = hs[0, l_idx:r_idx, :]
            if word_embeddings.shape[0] > 1:
                word_embeddings = word_embeddings.mean(0).unsqueeze(0)
            embeddings.append(word_embeddings)

        embeddings = torch.cat(embeddings)
        return embeddings


# class GPT2Embedder:
#     def __init__(self, layer: int, device: int = 0):

#         self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
#         self.model = AutoModel.from_pretrained("gpt2", output_hidden_states=True).to(
#             "cuda"
#         )
#         self.device = device

#         self.feature = NewFeatureExtractionPipeline(
#             layer=layer, model=self.model, tokenizer=self.tokenizer, device=device
#         )
#         self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

#     def get_single_embedding(self, word: str):

#         embedding_middle = self.feature([f" {word}"])
#         embedding_start = self.feature([f"{word}"])

#         embedding_middle = np.mean(embedding_middle[0][0], axis=0)
#         embedding_start = np.mean(embedding_start[0][0], axis=0)

#         return embedding_start, embedding_middle

#     def get_embedding_from_dataset(self, words: List[str], texts: List[str], **kwargs):
#         max_seq_length = kwargs.get("max_seq_length", 500)

#         dataset = self._tokenize_dataset(texts, max_seq_length)
#         dataset.set_format("pt")

#         features_from_model = self.feature(texts)

#         which_tokenization = []
#         for word, t in zip(words, texts):

#             if t.split()[0] == word:
#                 tok_word_start = self.tokenizer(word, add_special_tokens=False)
#                 which_tokenization.append(tok_word_start)
#             else:
#                 tok_word_middle = self.tokenizer(f" {word}", add_special_tokens=False)
#                 which_tokenization.append(tok_word_middle)

#         idx = [
#             find_sub_list(tok_word["input_ids"], input_ids.tolist())[0]
#             for tok_word, input_ids in zip(which_tokenization, dataset["input_ids"])
#         ]

#         # average over sub tokens
#         embeddings = list()

#         for hs, (l_idx, r_idx) in zip(features_from_model, idx):
#             word_embeddings = hs[0][l_idx:r_idx]
#             if len(word_embeddings) > 1:
#                 word_embeddings = np.mean(word_embeddings, axis=0)
#             embeddings.append(word_embeddings)

#         return embeddings

#     def _tokenize_dataset(self, texts, max_seq_length):
#         d = {"text": texts}
#         dataset = Dataset.from_dict(d)

#         def tokenize_text(examples):
#             return self.tokenizer(
#                 examples["text"],
#                 padding="max_length",
#                 truncation=True,
#                 max_length=max_seq_length,
#             )

#         try:
#             datasets.set_progress_bar_enabled(False)
#         except:
#             datasets.logging.disable_progress_bar()
#         # tokenize the corpus
#         dataset = dataset.map(
#             tokenize_text, batched=True, desc="Tokenizing", remove_columns=["text"]
#         )
#         return dataset
