from collections import defaultdict
from typing import List

import datasets
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import *


class ContextualizedEmbedder:
    def __init__(self, model_name: str, max_length: int, device: str = "cuda"):
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(
            model_name, output_hidden_states=True
        ).to(device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.device = device

    def find_sub_list(self, sl, l):
        results = list()
        sll = len(sl)

        for ind in (i for i, e in enumerate(l) if e == sl[0]):
            if l[ind : ind + sll] == sl:
                results.append((ind, ind + sll))

        return results

    def subset_of_tokenized(self, list_of_tokens: List[int]):
        """
        Takes in input the tokens that build a word + the padding
        Removes all the padding and returns only the ids of the word token
        """
        idx = list_of_tokens.index(self.tokenizer.pad_token_id)
        return list_of_tokens[0:idx]

    @torch.no_grad()
    def embed(
        self,
        words: List[str],
        target_texts: List[str],
        layers_id: List[int],
        batch_size: int = 1,
        show_progress: bool = True,
        *,
        averaging: bool = False,
        return_static: bool = False,
        add_special_tokens: bool = True,
    ):
        """Generate contextualized embeddings of words in contexts.

        Args:
            words List[int]: list of words to extract contextualized embeddings of
            layers_id List[int]: layers of interest
            target_texts List[str]: list of texts to use as contexts. Use the same "words" to get non-contextualised embeddings
            batch_size (int): Default: 1
            show_progress (bool): enable or disable progress bars. Default: True
            averaging (bool): if words are composed of sub-tokens, return the average between them. If set to false, we use the embedding of the
                first sub-token. Default: False
            return_static (bool): returns the static word embedding before positional and token_type summation. Default: False
            add_special_tokens (bool): include or not special tokens when embedding target_texts

        Returns:
            Dict[int, List[numpy.array]]: for each integer in 'layers_id', return a list of numpy arrays each corresponding to the contextualized
                embedding of the word in 'words' at that layer. When computed, static word embeddings, have index -1.

        """
        if (not words) or (not layers_id) or (not target_texts):
            raise ValueError("words, target_texts, and layers_id cannot be None")

        words = [f" {word.strip()}" for word in words]
        embs = defaultdict(list)

        # Extract embeddings
        original = pd.DataFrame({"text": target_texts, "words": words})
        test_dataset = datasets.Dataset.from_pandas(original)

        def tokenizer_function(examples):
            text_inputs = self.tokenizer(
                examples["text"],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                add_special_tokens=add_special_tokens,
            )
            word_inputs = self.tokenizer(
                examples["words"],
                max_length=20,
                padding="max_length",
                truncation=True,
                add_special_tokens=False,
            )

            examples["input_ids"] = text_inputs.input_ids
            examples["attention_mask"] = text_inputs.attention_mask
            examples["words_input_ids"] = word_inputs.input_ids
            return examples

        encoded_test = test_dataset.map(
            tokenizer_function,
            remove_columns=["text", "words"],
            desc="Text tokenization",
        )
        encoded_test.set_format("pt")

        dl = DataLoader(
            encoded_test, batch_size=batch_size, shuffle=False, pin_memory=True
        )

        pbar = tqdm(total=len(dl), position=0, disable=not show_progress)

        for batch in dl:

            words_ids = batch["words_input_ids"]
            pbar.update(1)
            del batch["words_input_ids"]

            assert words_ids.shape[0] == batch["input_ids"].shape[0]

            batch = {k: v.to(self.device) for k, v in batch.items()}

            features = self.model(**batch)["hidden_states"]

            for layer in layers_id:
                layer_features = features[layer]

                try:
                    idx = [
                        self.find_sub_list(
                            self.subset_of_tokenized(tok_word.tolist()),
                            input_ids.tolist(),
                        )[0]
                        for tok_word, input_ids in zip(words_ids, batch["input_ids"])
                    ]
                except IndexError as e:
                    raise Exception(
                        "Index Error: do all the words occur in the respective sentences?"
                    )

                if averaging:
                    for embedded_sentence_tokens, (l_idx, r_idx) in zip(
                        layer_features, idx
                    ):
                        embs[layer].append(
                            embedded_sentence_tokens[l_idx:r_idx, :]
                            .mean(0)
                            .detach()
                            .cpu()
                            .numpy()
                        )
                else:
                    for embedded_sentence_tokens, (l_idx, r_idx) in zip(
                        layer_features, idx
                    ):
                        # we use only the first sub-token
                        embs[layer].append(
                            embedded_sentence_tokens[l_idx : l_idx + 1, :]
                            .mean(0)
                            .detach()
                            .cpu()
                            .numpy()
                        )

            if return_static:
                word_embeddings = self.model.get_input_embeddings()

                for tok_word in words_ids:
                    w_ids = self.subset_of_tokenized(tok_word.tolist())
                    w_embs = word_embeddings(torch.LongTensor(w_ids).to(self.device))

                    if averaging:
                        embs[-1].append(w_embs.mean(0).detach().cpu().numpy())
                    else:
                        embs[-1].append(w_embs[0].detach().cpu().numpy())

        pbar.close()

        return embs
