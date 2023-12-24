import sys
from pathlib import Path

import qdrant_client
from llama_index import (
    ServiceContext,
    VectorStoreIndex,
    download_loader,
)
from llama_index.llms import Ollama
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore


def main(query):
    # Load local PDF file
    PDFReader = download_loader("PDFReader")
    loader = PDFReader()
    documents = loader.load_data(Path("./data/cv.pdf"))

    # Initialize Qdrant client
    client = qdrant_client.QdrantClient(path="./qdrant_data")
    vector_store = QdrantVectorStore(client=client, collection_name="tweets")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Setup service context
    llm = Ollama(model="llama2")
    service_context = ServiceContext.from_defaults(llm=llm, embed_model="local")

    # Build index
    index = VectorStoreIndex.from_documents(
        documents, service_context=service_context, storage_context=storage_context
    )

    # Query
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    print(response)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide a query as an argument.")
    else:
        query = sys.argv[1]
        main(query)
