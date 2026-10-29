# Retrieval-augmented generation

RAG gives a model facts it wasn't trained on by retrieving relevant documents
at question time and putting them in the prompt.

- Chunking: split documents into passages small enough to retrieve precisely.
- Embeddings: map chunks and queries into the same vector space.
- Retrieval quality matters more than the generator: bad chunks in, bad answers out.
