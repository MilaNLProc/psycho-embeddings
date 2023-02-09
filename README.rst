=================
Psycho Embeddings
=================


.. .. image:: https://img.shields.io/pypi/v/psycho_embeddings.svg
..         :target: https://pypi.python.org/pypi/psycho_embeddings

.. .. image:: https://img.shields.io/travis/MilaNLProc/psycho_embeddings.svg
..         :target: https://travis-ci.com/MilaNLProc/psycho_embeddings

.. .. image:: https://readthedocs.org/projects/psycho-embeddings/badge/?version=latest
..         :target: https://psycho-embeddings.readthedocs.io/en/latest/?version=latest
..         :alt: Documentation Status


A Python package to extract contextualised embeddings.


Getting Started
---------------

We require a python environment with a fully functional PyTorch installation. Then, please install our dependencies with:

.. code-block:: bash

    pip install -r requirements.txt


Examples
--------

**Extract the contextualised embedding of words in context** \
You can request representations:

* for one or more layers (`layers_id`)
* including static non-contextualised vectors (`return_static`)


.. code-block:: python
    from psycho_embeddings import ContextualizedEmbedder

    model = ContextualizedEmbedder("bert-base-cased")

    embeddings = model.embed(
        words=["play", "play"],
        target_texts=["I like the way you play.", "The play was outstanding."],
        layers_id=range(13),
        batch_size=8,
        return_static=True,
    )

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
