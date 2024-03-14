"""
    Generate query-answer-context JSON files for one model.
    Input are the queries from the jung_und_naiv.json file.
"""

import json
import os
import random

import torch
import transformers
from llama_index import ServiceContext, set_global_service_context, StorageContext, load_index_from_storage, \
    VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms import HuggingFaceLLM
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import DOCUMENT_COLLECTIONS_DIR, VECTOR_STORES_DIR, EVALUATION_GENERATED_ANSWERS_JUNG_UND_NAIV_DIR, \
    EVALUATION_QUERYSETS_DIR, EVALUATION_GENERATED_ANSWERS_WIKIPEDIA_DIR

random.seed(24)
transformers.logging.set_verbosity_error()

# archive_name = "jung_und_naiv"
archive_name = "wikipedia_evaluation"

all_documents = SimpleDirectoryReader(os.path.join(DOCUMENT_COLLECTIONS_DIR, archive_name)).load_data()

# dataset_json = "jung_und_naiv.json"
dataset_json = "wikipedia_evaluation.json"

model_ids = [
    "mistralai/Mistral-7B-Instruct-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "Intel/neural-chat-7b-v3-1",
    "meta-llama/Llama-2-7b-chat-hf",
]

json_names = [
    "mistral_instruct_v01.json",
    "mistral_instruct_v02.json",
    "neural_chat.json",
    "llama2_chat.json",
]

simple_names = [
    "mistral_instruct_v01",
    "mistral_instruct_v02",
    "neural_chat",
    "llama2_chat",
]

with open(os.path.join(EVALUATION_QUERYSETS_DIR, dataset_json), 'r') as file:
    querysets = json.load(file)

query_sets = [d for d in querysets["examples"]]
sampled_queries = random.sample(query_sets, 100)

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Couldn't iterate over models due to CUDA memory not being properly released after finishing with each model.
model_num = 0
model_id = model_ids[model_num]
json_name = json_names[model_num]
simple_name = simple_names[model_num]

responses = []

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype).to(device)
if model.config.pad_token_id is None:
    model.config.pad_token_id = model.config.eos_token_id
tokenizer = AutoTokenizer.from_pretrained(model_id)
hf_llm = HuggingFaceLLM(
    model=model,
    model_name=model_id,
    tokenizer=tokenizer,
    max_new_tokens=1024,
    generate_kwargs={
        "temperature": 0.3,
        "do_sample": True,
        "repetition_penalty": 1.1,
        "num_return_sequences": 1,
    },
    device_map="auto",
    model_kwargs={"torch_dtype": torch_dtype}
)
service_context = ServiceContext.from_defaults(
    llm=hf_llm,
    embed_model="local:BAAI/bge-base-en-v1.5",
)
set_global_service_context(service_context)

try:
    st_cont = StorageContext.from_defaults(persist_dir=os.path.join(VECTOR_STORES_DIR, simple_name))
    vector_index = load_index_from_storage(st_cont)
except:
    vector_index = VectorStoreIndex.from_documents(all_documents, service_context=service_context, show_progress=True)
    vector_index.storage_context.persist(persist_dir=os.path.join(VECTOR_STORES_DIR, simple_name))

query_engine = vector_index.as_query_engine()

for idx, query_set in enumerate(sampled_queries):
    query = query_set["query"]
    response = query_engine.query(query)
    contexts = [n.text for n in response.source_nodes]
    print(f"[{idx} / {len(sampled_queries)}] Finished running Query: {query}")

    responses.append({"query": query, "response": response.response, "contexts": contexts})

with open(os.path.join(EVALUATION_GENERATED_ANSWERS_WIKIPEDIA_DIR, json_name), 'w') as fout:
    json.dump(responses, fout)
