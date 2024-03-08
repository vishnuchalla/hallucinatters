from llama_index.core import ServiceContext, PromptHelper, StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.faiss import FaissVectorStore
import tiktoken
import os
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['OPENAI_API_KEY'] = "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"

import faiss

# dimensions of text-ada-embedding-002
#d = 1536
d=768
faiss_index = faiss.IndexFlatL2(d)

llm = OpenAI(model='gpt-3.5-turbo', temperature=0, max_tokens=256)

embed_model = HuggingFaceEmbedding(model_name="google-bert/bert-base-uncased")


node_parser = SimpleNodeParser.from_defaults(
  separator=" ",
  chunk_size=1024,
  chunk_overlap=20,
  #tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode
)

prompt_helper = PromptHelper(
  context_window=4096, 
  num_output=256, 
  chunk_overlap_ratio=0.1, 
  chunk_size_limit=None
)

service_context = ServiceContext.from_defaults(
  llm=None,
  embed_model=embed_model,
  node_parser=node_parser,
  prompt_helper=prompt_helper
)

documents = SimpleDirectoryReader(input_dir='hallucinators_rag').load_data()

vector_store = FaissVectorStore(faiss_index=faiss_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents, 
    service_context = service_context,
    storage_context=storage_context,
    show_progress=True
    )

index.storage_context.persist(persist_dir="hfpersist")

query_engine = index.as_query_engine(service_context=service_context)
response = query_engine.query("What is tuition reimbursement?")
print(response)