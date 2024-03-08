from llama_index.core import ServiceContext, PromptHelper, StorageContext, load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings,get_response_synthesizer
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
import tiktoken
import os
from collections import defaultdict 

os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['OPENAI_API_KEY'] = "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"
vector_store = FaissVectorStore.from_persist_dir("./ragpersist")
storage_context = StorageContext.from_defaults(
    vector_store=vector_store, persist_dir="./ragpersist"
)
index = load_index_from_storage(storage_context=storage_context)
llm = OpenAI(model='gpt-3.5-turbo', temperature=0, max_tokens=256)
embed_model = OpenAIEmbedding()


node_parser = SimpleNodeParser.from_defaults(
  separator=" ",
  chunk_size=1024,
  chunk_overlap=20,
  tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode
)

prompt_helper = PromptHelper(
  context_window=4096, 
  num_output=256, 
  chunk_overlap_ratio=0.1, 
  chunk_size_limit=None
)


Settings.llm=None
Settings.embed_model=embed_model
Settings.node_parser=node_parser
Settings.prompt_helper=prompt_helper

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
#query_engine = index.as_query_engine(service_context=service_context)
response = query_engine.query("What is ESPP? what is the discount")
file_dict=dict()
for context in response.source_nodes:
    file_path = context.metadata.get('file_path')
    if file_path not in file_dict:
        file_dict[file_path] = context.get_text()
    else:
        file_dict[file_path] = file_dict[file_path] + "\n\n" + context.get_text()
print("\n\n".join(file_dict.values()))