# DocuMentor by Florian Tobias Paul, Lena Anna Klosik and Florian Freiberger
This is a study project made for the module "Advanced topics in AI" by Prof. Osendorfer at the HAW Landshut.

It is a Tool which allows users to ask questions about their documents.
These documents can be uploaded within the interface.
Possible document types include:
- .txt files
- .mp3 files
- .mp4 files
- Links to Youtube Videos or Playlists

After choosing an archive and adding documents to it, the user can then ask the chatbot about it.
Archives are stored server-sided and can still be accessed after restarting.

## Setup
- Create your Conda environment:  
        `conda env create -f llm_studienprojekt_environment.yml`
- Activate the environment:
        `conda activate llm_env`  
- Make sure that the GPU is compatible with CUDA (Running on CPU is not supported due to extreme processing times)  
- Start main.py  

## Used Tools
- Gradio (Interface)
- Llama-Index (Indexing and retrieving from documents)
- OpenAI Whisper (Transcribing audio files)
- Multiple Huggingface LLMs (Chatting with the user):
    - mistralai/Mistral-7B-Instruct-v0.1
    - **mistralai/Mistral-7B-Instruct-v0.2** (Selected for the final chatbot)
    - Intel/neural-chat-7b-v3-1
    - meta-llama/Llama-2-7b-chat-hf

## Project Structure
llm_studienprojekt/  
│  
├── chatbot_gradio/  
│ └── chatbot_gradio.py  
│  
├── chatbot_helper_functions/  
│ ├── temp_files/  
│ ├── custom_condense_plus_context.py  
│ ├── logger.py  
│ ├── model_operations.py  
│ ├── save_input_files.py  
│ ├── transcription.py  
│ └── youtube_download.py    
│  
├── evaluation/  
│ ├── generated_answers/  
│ │ ├── jung_und_naiv/  
│ │ └── wikipedia_evaluation/  
│ ├── querysets/  
│ ├── results/  
│ │ ├── jung_und_naiv/  
│ │ └── wikipedia_evaluation/  
│ ├── answers_generation.py  
│ ├── dataset_generation.py  
│ ├── ealuation.py  
│ └── result_evaluation.py  
│  
├── evaluation_chat_engine/  
│ └── chat_engine_test.ipynb  
│
├── file_storage/  
│ ├── document_collections/  
│ ├── mp3s/  
│ └── mp4s/  
│  
├── vector_stores/  
│  
├── config.py  
│  
├── **main.py** (Start here)  
│  
└── README.md  
