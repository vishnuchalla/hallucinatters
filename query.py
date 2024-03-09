import os
import argparse
import warnings

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import Settings, load_index_from_storage
from langchain_community.llms import HuggingFaceTextGenInference
from llama_index.core import PromptHelper, StorageContext, Settings,get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from langchain_core._api.deprecation import LangChainDeprecationWarning
import requests
requests.packages.urllib3.disable_warnings()

warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


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
    print(args)

    print("Preparing node parser")
    node_parser = SimpleNodeParser.from_defaults(
    separator=" ",
    chunk_size=args.chunk,
    chunk_overlap=args.overlap,
    )

    print("Initializing prompt helper")
    prompt_helper = PromptHelper(
    context_window=4096, 
    num_output=256, 
    chunk_overlap_ratio=0.1, 
    chunk_size_limit=None
    )

    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    os.environ['OPENAI_API_KEY'] = "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"
    embed_model = args.model if args.model else OpenAIEmbedding()
    Settings.llm=None
    Settings.embed_model=embed_model
    Settings.node_parser=node_parser
    Settings.prompt_helper=prompt_helper

    print("Setting up storage context for index load...")
    vector_store = FaissVectorStore.from_persist_dir(args.persist_dir)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, persist_dir=args.persist_dir
    )

    print("Loading up the index")
    index = load_index_from_storage(
        index_id=args.index_name,
        storage_context=storage_context 
    )

    print("Fetching similar docs and summarizing them")
    ret= VectorIndexRetriever(
        index=index,
        similarity_top_k=4
    )
    response_syn=get_response_synthesizer(
        response_mode="tree_summarize",
    )
    query_engine= RetrieverQueryEngine(
        retriever=ret,
        response_synthesizer=response_syn,
    )
    response = query_engine.query(args.query)
    file_dict=dict()
    for context in response.source_nodes:
        file_path = context.metadata.get('file_path')
        if file_path not in file_dict:
            file_dict[file_path] = context.get_text()
        else:
            file_dict[file_path] = file_dict[file_path] + "\n\n" + context.get_text()
    context = "\n\n".join(file_dict.values())

    server_url = "https://llama-2-7b-chat-perfconf-hackathon.apps.dripberg-dgx2.rdu3.labs.perfscale.redhat.com"
    llm = HuggingFaceTextGenInference(
        inference_server_url=server_url,
        max_new_tokens=2000,
        top_k=10,
        top_p=0.95,
        typical_p=0.95,
        temperature=0.7,
        repetition_penalty=1.03,
    )

    prompt_template = """
    ### QUESTION:
    {question}

    ### ANSWER:
    """

    prompt = PromptTemplate(
        input_variables=["question"],
        template=prompt_template,
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    answer = llm_chain.invoke(args.query)

    print(answer)

    