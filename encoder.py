import os
import time

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.storage.storage_context import StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss

import argparse
import asyncio

# Constant
INDEX_NAME = "hallucinatters_rag_index"

async def main():

    start_time = time.time()
    parser = argparse.ArgumentParser(
        description="embedding cli for task execution"
    )
    parser.add_argument("-t", "--vector-type",   default="local", help="Type of vector db [local,faiss]")
    parser.add_argument("-c", "--chunk",   default="1500", help="Chunk size for embedding")
    parser.add_argument("-l", "--overlap",   default="10", help="Chunk overlap for embedding")
    parser.add_argument("-f", "--folder", help="Plain text folder path")
    parser.add_argument("-m", "--model",   default="embeddings_model", help="LLM model used for embeddings [local, llama2, or any other supported by llama_index]")
    parser.add_argument("-o", "--output", help="Vector DB output folder")


    # execute
    args = parser.parse_args()  

    print(args)

    PERSIST_FOLDER = args.output

    # setup storage context
    match args.vector_type:
        case "local":
            print("** Local embeddings")
            storage_context = StorageContext.from_defaults()
        case "faiss":
            faiss_index = faiss.IndexFlatL2(768)
            vector_store = FaissVectorStore(faiss_index=faiss_index)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("** Configured storage context")
    documents = SimpleDirectoryReader(input_dir=args.folder, recursive=True).load_data()
    print("** Loading docs ")
    Settings.chunk_size = int(args.chunk)
    Settings.chunk_overlap = int(args.overlap)
    Settings.embed_model = HuggingFaceEmbedding(model_name=args.model)
    Settings.llm = None
    os.environ["HF_HOME"] = args.model
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, show_progress=True)
    index.set_index_id(INDEX_NAME)
    index.storage_context.persist(persist_dir=PERSIST_FOLDER)
    print("*** Completed  embeddings ")

    end_time = time.time()
    execution_time_seconds = end_time - start_time

    print(f"** Total execution time in seconds: {execution_time_seconds}")

asyncio.run(main())
