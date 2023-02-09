# Psycho Embeddings

A Python package to extract contextualised embeddings.

## Getting Started

We require a python environment with a fully functional PyTorch
installation. Then, please install our dependencies with:

```bash
pip install -r requirements.txt
```

## Examples

**Extract the contextualised embedding of words in context** \

You can request representations:

-   for one or more layers (`layers_id`)
-   including static non-contextualised vectors (`return_static`)

```python
from psycho_embeddings import ContextualizedEmbedder
model = ContextualizedEmbedder("bert-base-cased")

embeddings = model.embed(
    words=["play", "play"],
    target_texts=["I like the way you play.", "The play was outstanding."],
    layers_id=range(13),
    batch_size=8,
    return_static=True,
)
```

## Credits

This package was created with
[Cookiecutter](https://github.com/audreyr/cookiecutter) and the
[audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage)
project template.
