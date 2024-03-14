import os.path

# Directory Paths
ROOT_DIRECTORY = os.path.dirname(__file__)

CHATBOT_GRADIO_DIR = os.path.join(ROOT_DIRECTORY, "chatbot_gradio")

CHATBOT_HELPER_FUNCTIONS_DIR = os.path.join(ROOT_DIRECTORY, "chatbot_helper_functions")
TEMP_FILE_STORAGE_DIR = os.path.join(CHATBOT_HELPER_FUNCTIONS_DIR, "temp_files")

EVALUATION_DIR = os.path.join(ROOT_DIRECTORY, "evaluation")
EVALUATION_GENERATED_ANSWERS_DIR = os.path.join(EVALUATION_DIR, "generated_answers")
EVALUATION_GENERATED_ANSWERS_JUNG_UND_NAIV_DIR = os.path.join(EVALUATION_GENERATED_ANSWERS_DIR, "jung_und_naiv")
EVALUATION_GENERATED_ANSWERS_WIKIPEDIA_DIR = os.path.join(EVALUATION_GENERATED_ANSWERS_DIR, "wikipedia_evaluation")
EVALUATION_RESULTS_DIR = os.path.join(EVALUATION_DIR, "results")
EVALUATION_RESULTS_JUNG_UND_NAIV_DIR = os.path.join(EVALUATION_RESULTS_DIR, "jung_und_naiv")
EVALUATION_RESULTS_WIKIPEDIA_DIR = os.path.join(EVALUATION_RESULTS_DIR, "wikipedia_evaluation")
EVALUATION_QUERYSETS_DIR = os.path.join(EVALUATION_DIR, "querysets")

FILE_STORAGE_DIR = os.path.join(ROOT_DIRECTORY, "file_storage")
DOCUMENT_COLLECTIONS_DIR = os.path.join(FILE_STORAGE_DIR, "document_collections")
MP3S_DIR = os.path.join(FILE_STORAGE_DIR, "mp3s")
MP4S_DIR = os.path.join(FILE_STORAGE_DIR, "mp4s")

VECTOR_STORES_DIR = os.path.join(ROOT_DIRECTORY, "vector_stores")

# DEFAULTS
DEFAULT_ARCHIVE_NAME = "jung_und_naiv"
DEFAULT_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
