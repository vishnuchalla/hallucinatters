import os
import argparse
import warnings
from RAG import Rag
from langchain_core._api.deprecation import LangChainDeprecationWarning
import requests
requests.packages.urllib3.disable_warnings()

warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['KMP_DUPLICATE_LIB_OK']='True'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="embedding cli for task execution"
    )
    parser.add_argument("-p", "--persist-dir", help="Vector DB persist dir")
    parser.add_argument("-c", "--chunk",   default="1024", help="Chunk size for embedding")
    parser.add_argument("-l", "--overlap",   default="10", help="Chunk overlap for embedding")
    parser.add_argument("-q", "--query", help="Query to ask the DB")
    parser.add_argument("-i", "--index-name", default="hallucinatters_rag_index", help="Vector DB persist dir")
    parser.add_argument("-m", "--model", help="LLM model used for embeddings [local, llama2, or any other supported by llama_index]")

    # execute
    args = parser.parse_args()

    rag=Rag(args=args)
    rag.queryRag(args.query)    
