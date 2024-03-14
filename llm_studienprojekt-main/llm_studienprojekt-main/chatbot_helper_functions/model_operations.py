import os

import torch
from llama_index import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage, \
    ServiceContext, set_global_service_context
from llama_index.llms import HuggingFaceLLM
from transformers import AutoModelForCausalLM, AutoTokenizer

from chatbot_helper_functions.custom_condense_plus_context import CustomCondensePlusContextChatEngine
from config import DOCUMENT_COLLECTIONS_DIR, VECTOR_STORES_DIR

GENERATE_KWARGS = {
    "temperature": 0.3,
    "do_sample": True,
    "repetition_penalty": 1.1,
    "num_return_sequences": 1,
}
MAX_NEW_TOKENS = 512


def get_vector_index(archive_name, service_context, first_start=False):
    """
    Gets a Vector Store or Creates a new one if one with the given Archive Name does not exist yet.

    :param archive_name: Name of the archive
    :param service_context: Service Context
    :return:
    """
    set_global_service_context(service_context)
    archive_path = os.path.join(DOCUMENT_COLLECTIONS_DIR, archive_name)
    if not os.path.exists(archive_path):
        print(f"No Documents Directory for Archive \"{archive_name}\" found.")
        print(f"Creating new Collections Directory...\n")
        os.makedirs(archive_path)
        with open(os.path.join(str(archive_path), "first_textfile.txt"), 'w'):
            pass
        print(f"Created new Document Collections Directory for Archive \"{archive_name}\".\n")
    documents = SimpleDirectoryReader(str(archive_path)).load_data()
    print(f"Searching for Vector Index for Archive \"{archive_name}\".")
    try:
        st_cont = StorageContext.from_defaults(persist_dir=str(os.path.join(VECTOR_STORES_DIR, archive_name)))
        vector_index = load_index_from_storage(st_cont)
        print(f"Vector Index for Archive \"{archive_name}\" found.\n")
        if not first_start:
            print(f"Refreshing Vector Index (This can take a few minutes for large archives!) ...")
            vector_index.refresh(documents)
            print(f"Refreshed Vector Index.\n")
    except:
        print(f"Vector Index for Archive \"{archive_name}\" not found. Creating new Vector Index...")
        if not os.path.exists(archive_path):
            os.makedirs(archive_path)
        vector_index = VectorStoreIndex.from_documents(documents, service_context=service_context, show_progress=True)
        vector_index.storage_context.persist(persist_dir=str(os.path.join(VECTOR_STORES_DIR, archive_name)))
        print(f"Created new Vector Index for Archive \"{archive_name}\".\n")

    return vector_index


def init_service_context(model_id):
    print(f"Initializing Service Context...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype).to(device)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    hf_llm = HuggingFaceLLM(
        model=model,
        model_name=model_id,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        generate_kwargs=GENERATE_KWARGS,
        device_map="auto",
        model_kwargs={"torch_dtype": torch_dtype},
        is_chat_model=True,
    )
    service_context = ServiceContext.from_defaults(
        llm=hf_llm,
        embed_model="local:BAAI/bge-base-en-v1.5",
    )
    print(f"Initialized Service Context successfully.")
    return service_context, model


def init_chat_engine(service_context, retriever):
    print("Initializing Chat Engine...")
    chat_engine = CustomCondensePlusContextChatEngine.from_defaults(
        service_context=service_context,
        skip_condense=True,
        retriever=retriever,
    )
    print("Initialized Chat Engine successfully.\n")
    return chat_engine


def refresh_vector_index(vector_index, documents):
    """
    Refreshes the Vector Index, which makes it add new and changed documents
    """
    vector_index.refresh(documents)

    return
