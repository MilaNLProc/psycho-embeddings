=================
Psycho Embeddings
=================


.. image:: https://img.shields.io/pypi/v/psycho_embeddings.svg
        :target: https://pypi.python.org/pypi/psycho_embeddings

.. image:: https://img.shields.io/travis/MilaNLProc/psycho_embeddings.svg
        :target: https://travis-ci.com/MilaNLProc/psycho_embeddings

.. image:: https://readthedocs.org/projects/psycho-embeddings/badge/?version=latest
        :target: https://psycho-embeddings.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status




Psycho Embeddings


* Free software: MIT license
* Documentation: https://psycho-embeddings.readthedocs.io.


Features
--------

Single Sentence Embedding

.. code-block:: python

    gpt2 = GPT2Embedder()

    embedding = gpt2.get_single_embedding("I like the play", "play", layer=12)

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
