import os
import random

import torch
import transformers
from llama_index import ServiceContext, SimpleDirectoryReader, Prompt, set_global_service_context
from llama_index.llama_dataset.generator import RagDatasetGenerator
from llama_index.llms import HuggingFaceLLM
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import DOCUMENT_COLLECTIONS_DIR, EVALUATION_QUERYSETS_DIR

transformers.logging.set_verbosity_error()

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "mistralai/Mistral-7B-Instruct-v0.2"
# json_name = "jung_und_naiv.json"
json_name = "wikipedia_evaluation.json"
# archive_name = "jung_und_naiv"
archive_name = "wikipedia_evaluation"

all_documents = SimpleDirectoryReader(os.path.join(DOCUMENT_COLLECTIONS_DIR, archive_name)).load_data()
# document_samples_amount = 24
document_samples_amount = 12
sampled_documents = random.sample(all_documents, document_samples_amount)

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)
if model.config.pad_token_id is None:
    model.config.pad_token_id = model.config.eos_token_id  # This is not working? So Verbosity was set to Error.
hf_llm = HuggingFaceLLM(
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    model_kwargs={"torch_dtype": torch_dtype}
)
service_context = ServiceContext.from_defaults(
    llm=hf_llm,
    embed_model="local:BAAI/bge-base-en-v1.5"
)
set_global_service_context(service_context)

text_question_template_jun = Prompt(
    "A sample from an interview by the YouTube Channel \"Jung und Naiv\" is given below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Using the interview sample, carefully follow the instructions below:\n"
    "{query_str}"
)
text_question_template_wiki = Prompt(
    "A sample from an interview by the YouTube Channel \"Jung und Naiv\" is given below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Using the interview sample, carefully follow the instructions below:\n"
    "{query_str}"
)

dataset_generator_rag = RagDatasetGenerator.from_documents(
    documents=sampled_documents,
    num_questions_per_chunk=1,
    service_context=service_context,
    show_progress=True,
    text_question_template=text_question_template_wiki,
    question_gen_query=(
        "You are an evaluator for a search pipeline. Your task is to write a single question "
        "using the provided documentation sample above to test the search pipeline. The question could "
        "reference specific names, facts, information or events. Restrict the question to the context information "
        "provided. When referring to individuals or specific entities or events, use their full names or a detailed "
        "designation rather than pronouns or general titles. It should be understandable who or what is referred to "
        "even without the direct context. Keep the question concise.\n"
        "Question: "
    ),
)

print(len(dataset_generator_rag.nodes))

generated_dataset = dataset_generator_rag.generate_dataset_from_nodes()
generated_dataset.save_json(os.path.join(EVALUATION_QUERYSETS_DIR, json_name))
