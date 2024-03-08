import argparse

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, load_index_from_storage
from llama_index.core.storage.storage_context import StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="embedding cli for task execution"
    )
    parser.add_argument("-p", "--persist-dir", help="Vector DB persist dir")
    parser.add_argument("-q", "--query", help="Query to ask the DB")
    parser.add_argument("-i", "--index-name", default="hallucinatters_rag_index", help="Vector DB persist dir")
    parser.add_argument("-m", "--model",   default="embeddings_model", help="LLM model used for embeddings [local, llama2, or any other supported by llama_index]")

    # execute
    args = parser.parse_args()
    print(args)

    Settings.embed_model = HuggingFaceEmbedding(model_name=args.model)
    Settings.llm = None

    print("Setting up storage context for index load...")

    vector_store = FaissVectorStore.from_persist_dir(args.persist_dir)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, persist_dir=args.persist_dir
    )


    index = load_index_from_storage(
        index_id=args.index_name,
        storage_context=storage_context 
    )

    response = index.as_query_engine().query(args.query)
    print(response)