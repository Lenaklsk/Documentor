import json
import os
import random

import torch
import transformers
from llama_index import ServiceContext, set_global_service_context, SimpleDirectoryReader, ChatPromptTemplate, \
    PromptTemplate
from llama_index.core.llms.types import MessageRole, ChatMessage
from llama_index.evaluation import FaithfulnessEvaluator, CorrectnessEvaluator
from llama_index.evaluation.correctness import DEFAULT_SYSTEM_TEMPLATE, DEFAULT_USER_TEMPLATE, DEFAULT_EVAL_TEMPLATE
from llama_index.llms import HuggingFaceLLM
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import DOCUMENT_COLLECTIONS_DIR, EVALUATION_GENERATED_ANSWERS_JUNG_UND_NAIV_DIR, \
    EVALUATION_GENERATED_ANSWERS_WIKIPEDIA_DIR, EVALUATION_RESULTS_JUNG_UND_NAIV_DIR, \
    EVALUATION_RESULTS_WIKIPEDIA_DIR

transformers.logging.set_verbosity_error()

cwd = os.path.abspath(os.path.dirname(__file__))
# all_documents = SimpleDirectoryReader(os.path.join(DOCUMENT_COLLECTIONS_DIR, "jung_und_naiv")).load_data()
all_documents = SimpleDirectoryReader(os.path.join(DOCUMENT_COLLECTIONS_DIR, "wikipedia_evaluation")).load_data()
random.seed(24)

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

eval_model_id = "mistralai/Mistral-7B-Instruct-v0.2"
answers_filenames = [
    "mistral_instruct_v01",
    "mistral_instruct_v02",
    "llama2_chat",
    "neural_chat",
]

results = []
faith_results = []
correctness_results = []

model = AutoModelForCausalLM.from_pretrained(eval_model_id, torch_dtype=torch_dtype).to(device)
if model.config.pad_token_id is None:
    model.config.pad_token_id = model.config.eos_token_id
evaluator_tokenizer = AutoTokenizer.from_pretrained(eval_model_id)
evaluator_hf_llm = HuggingFaceLLM(
    model=model,
    model_name=eval_model_id,
    tokenizer=evaluator_tokenizer,
    max_new_tokens=128,
    generate_kwargs={
        "temperature": 0.3,
        "do_sample": True,
        "repetition_penalty": 1.1,
        "num_return_sequences": 1,
    },
    device_map="auto",
    model_kwargs={"torch_dtype": torch_dtype},
    is_chat_model=True,
)
service_context = ServiceContext.from_defaults(
    llm=evaluator_hf_llm,
    embed_model="local:BAAI/bge-base-en-v1.5",
)
set_global_service_context(service_context)

if "instruct" in eval_model_id.lower():
    eval_template = ChatPromptTemplate(
        message_templates=[
            ChatMessage(role=MessageRole.USER, content=DEFAULT_SYSTEM_TEMPLATE),
            ChatMessage(role=MessageRole.ASSISTANT, content="Understood, I will follow these Instructions!"),
            ChatMessage(role=MessageRole.USER, content=DEFAULT_USER_TEMPLATE),
        ]
    )
else:
    eval_template = DEFAULT_EVAL_TEMPLATE

faith_eval_template = PromptTemplate(
    "Please tell if a given piece of information "
    "is supported by the context.\n"
    "You need to answer with either YES or NO.\n"
    "Answer YES and only YES if any of the context supports the information, even "
    "if most of the context is unrelated. "
    "Answer NO and only NO if none of the context supports the information."
    "Your answer can only be YES or NO, without further text."
    "Some examples are provided below. \n\n"
    "Information: Apple pie is generally double-crusted.\n"
    "Context: An apple pie is a fruit pie in which the principal filling "
    "ingredient is apples. \n"
    "Apple pie is often served with whipped cream, ice cream "
    "('apple pie à la mode'), custard or cheddar cheese.\n"
    "It is generally double-crusted, with pastry both above "
    "and below the filling; the upper crust may be solid or "
    "latticed (woven of crosswise strips).\n"
    "Answer: YES\n"
    "Information: Apple pies tastes bad.\n"
    "Context: An apple pie is a fruit pie in which the principal filling "
    "ingredient is apples. \n"
    "Apple pie is often served with whipped cream, ice cream "
    "('apple pie à la mode'), custard or cheddar cheese.\n"
    "It is generally double-crusted, with pastry both above "
    "and below the filling; the upper crust may be solid or "
    "latticed (woven of crosswise strips).\n"
    "Answer: NO\n"
    "Information: {query_str}\n"
    "Context: {context_str}\n"
    "Answer: "
)

faith_evaluator = FaithfulnessEvaluator(service_context=service_context, eval_template=faith_eval_template)
correctness_evaluator = CorrectnessEvaluator(service_context=service_context, eval_template=eval_template)

for answers_filename in answers_filenames:
    vector_store_name = answers_filename

    with open(os.path.join(EVALUATION_GENERATED_ANSWERS_WIKIPEDIA_DIR, answers_filename + ".json"), 'r') as file:
        answers = json.load(file)

    for qacs_el in answers:
        query = qacs_el["query"]
        contexts = qacs_el["contexts"]
        response = qacs_el["response"]

        faith_result = faith_evaluator.evaluate(response=response, contexts=contexts)
        correctness_result = correctness_evaluator.evaluate(query=query, response=response)

        results.append({
            "query": query,
            "response": response,
            "source": "\n\n\n".join(contexts),
            "correctness_result": {
                "score": correctness_result.score,
                "feedback": correctness_result.feedback,
            },
            "faith_result": {
                "score": faith_result.score,
                "feedback": faith_result.feedback,
            },
        })

    with open(os.path.join(EVALUATION_RESULTS_WIKIPEDIA_DIR, answers_filename + "_results.json"), 'w') as file:
        json.dump(results, file)
