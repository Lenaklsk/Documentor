"""
    Main class for the Gradio Interface.
    This includes the interface itself as well as many helper functions which are triggered by the interface.
    As many helper functions as possible were moved to the chatbot_helper_functions directory.
"""

import gc
import os
import sys
import time

import gradio as gr
import torch
import transformers
from llama_index import set_global_service_context

from chatbot_helper_functions.logger import Logger, read_logs

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import DEFAULT_ARCHIVE_NAME, DEFAULT_MODEL_ID, VECTOR_STORES_DIR
from chatbot_helper_functions.model_operations import get_vector_index, init_service_context, init_chat_engine
from chatbot_helper_functions.save_input_files import save_mp3_files, save_mp4_files, save_text_files, save_vector_db
from chatbot_helper_functions.youtube_download import check_yt_url_type_and_process, download_urls, convert_to_mp3
from chatbot_helper_functions.transcription import get_list_of_files_to_transcribe, setup_model_for_transcription

transformers.logging.set_verbosity_error()
sys.stdout = Logger("output.log")

# Initialize global variables
service_context, model = init_service_context(DEFAULT_MODEL_ID)
model_cpu = None

set_global_service_context(service_context)
archive_names = [name for name in os.listdir(VECTOR_STORES_DIR) if
                 os.path.isdir(os.path.join(VECTOR_STORES_DIR, name))]
current_archive_name = DEFAULT_ARCHIVE_NAME
vector_index = get_vector_index(current_archive_name, service_context, first_start=True)
vector_index.as_chat_engine()
retriever = vector_index.as_retriever()
chat_engine = init_chat_engine(service_context, retriever)

chat_history = []


def move_model_to_cpu():
    """
    Moves the model from CUDA to CPU to make room for Whisper.

    :return:
    """
    global model
    global model_cpu

    model_cpu = model.to('cpu')
    model = None
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return


def move_model_to_gpu():
    """
    Moves the model back to CUDA once Whisper has finished

    :return:
    """
    global model
    global model_cpu

    model = model_cpu.to('cuda')
    model_cpu = None
    del model_cpu
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return


def send_user_message(user_message, history):
    """
    Adds the user message to the chat history and the chat window.

    :param user_message: Latest user message
    :param history: Chat history
    :return:
    """
    global chat_history
    chat_history = history
    return "", history + [[user_message, None]]


def respond(history):
    """
    Calls the chat engine.
    Streams the answer to the chat window.
    Saves the new answer to the chat history.

    :param history: Chat history
    """
    response = chat_engine.stream_chat(history[-1][0])
    history[-1][1] = ""
    for character in response.response_gen:
        history[-1][1] += character
        time.sleep(0.05)
        yield history
    global chat_history
    chat_history = history


def clear_history():
    """
    Clears the chat history and resets the chat engine.

    :return:
    """

    global chat_history

    chat_history = []
    chat_engine.reset()
    print("Cleared Chat History and reset Chat Engine successfully.")
    return None


def change_archive(archive_name):
    """
    Changes the archived.
    Also rebuilds the vector store and retriever and chat engine accordingly.

    :param archive_name: Name of the archive to be changed to
    :return: String confirming the change with the archive name
    """
    global vector_index, current_archive_name, service_context, retriever, chat_engine

    if archive_name not in archive_names:
        archive_names.append(archive_name)
    current_archive_name = archive_name
    vector_index = get_vector_index(archive_name, service_context, first_start=True)
    retriever = vector_index.as_retriever()
    chat_engine = init_chat_engine(service_context, retriever)
    print(f"Changed archive to {archive_name} successfully.\n")

    return f"Changed archive to {archive_name}.\n"


def get_or_create_vector_store(archive_name):
    global vector_index, retriever, chat_engine
    vector_index = get_vector_index(archive_name, service_context)
    retriever = vector_index.as_retriever()
    chat_engine = init_chat_engine(service_context, retriever)


def process_youtube_links(youtube_links):
    global current_archive_name
    download_and_process_youtube_links(youtube_links, current_archive_name)

    return f"Processed YouTube links: {youtube_links}"


def download_and_process_youtube_links(yt_links, archive_name):
    urls_array = [url.strip() for url in yt_links.split(', ')]
    check_yt_url_type_and_process(urls_array)

    download_urls(archive_name)
    convert_videos_to_audio(archive_name)


def process_mp4_files(mp4_files):
    global current_archive_name
    save_mp4_files(mp4_files, current_archive_name)
    convert_videos_to_audio(current_archive_name)

    return "Processed MP3 Files"


def convert_videos_to_audio(archive_name):
    convert_to_mp3(archive_name)
    transcribe_audio_files(archive_name)


def process_mp3_files(mp3_files):
    global current_archive_name
    print(mp3_files)
    save_mp3_files(mp3_files, current_archive_name)
    transcribe_audio_files(current_archive_name)

    return "Processed MP3 Files"


def transcribe_audio_files(archive_name):
    move_model_to_cpu()
    get_list_of_files_to_transcribe(archive_name)
    setup_model_for_transcription(archive_name)
    move_model_to_gpu()
    get_or_create_vector_store(archive_name)


def process_text_files(text_files):
    global current_archive_name
    save_text_files(text_files, current_archive_name)
    get_or_create_vector_store(current_archive_name)

    return f"Processed input: {text_file}"


def process_vector_database(vector_ddb_files, archive_name):
    global archive_names
    if (archive_name in archive_names) or (archive_name == ""):
        return f"Archive with name \"{archive_name}\" already exists!"
    save_vector_db(vector_ddb_files, archive_name)

    return "Processed Vector Database"


def update_dropdown_choices():
    global archive_names
    print("In update function")
    print(archive_names)
    return gr.Dropdown(label="Archive Selection", choices=archive_names, allow_custom_value=True, value=current_archive_name)


with gr.Blocks(title="DocuMentor") as app:
    gr.Markdown("# DocuMentor - Chat with your Documents")

    with gr.Row():
        with gr.Tab("Chatbot"):
            chatbot = gr.Chatbot()
            msg = gr.Textbox()
            clear = gr.Button("Clear")
            msg.submit(send_user_message, [msg, chatbot], [msg, chatbot]).then(
                respond, chatbot, chatbot
            )
            clear.click(clear_history, None, chatbot, queue=False)

    with gr.Row():
        with gr.Column():
            with gr.Tab("Archive Selection - Select an existing Archive from Dropdown or type a Name for a new one."):
                archive_dropdown = gr.Dropdown(label="Archive Selection", choices=archive_names,
                                               allow_custom_value=True,
                                               value=DEFAULT_ARCHIVE_NAME)
                with gr.Row():
                    output_text = gr.Textbox(label="Output Text", interactive=False)
                    update_dropdown_button = gr.Button("Update Dropdown Choices (After adding a new Archive)")
                submit_text = gr.Button("Switch to this Archive")
                submit_text.click(fn=change_archive, inputs=[archive_dropdown], outputs=output_text)
                update_dropdown_button.click(fn=update_dropdown_choices, inputs=None, outputs=archive_dropdown, queue=False)

        with gr.Column():
            with gr.Tab("YouTube Playlist/Links"):
                with gr.Row():
                    youtube_links = gr.Textbox(label="YouTube Links", lines=3,
                                               placeholder="Enter YouTube playlist link or multiple video links separated by commas")
                youtube_output = gr.Textbox(label="Processed Links Output")
                process_youtube = gr.Button("Process YouTube Links")
                process_youtube.click(process_youtube_links, [youtube_links], youtube_output)

            with gr.Tab("MP4-Files"):
                with gr.Row():
                    # gr.Interface
                    mp4_files = gr.File(label="Upload MP4 Files", file_count="multiple")
                mp4_output = gr.Textbox(label="MP4 Processing Output")
                process_mp3 = gr.Button("Process MP4 Files")
                process_mp3.click(process_mp4_files, [mp4_files], mp4_output)

            with gr.Tab("MP3-Files"):
                with gr.Row():
                    # gr.Interface
                    mp3_files = gr.File(label="Upload MP3 Files", file_count="multiple")
                mp3_output = gr.Textbox(label="MP3 Processing Output")
                process_mp3 = gr.Button("Process MP3 Files")
                process_mp3.click(process_mp3_files, [mp3_files], mp3_output)

            with gr.Tab("Text-Files"):
                with gr.Row():
                    # gr.Interface
                    text_file = gr.File(label="Upload Text File", file_count="multiple")
                text_file_output = gr.Textbox(label="Processed Text Output")
                process_text_file = gr.Button("Process Input")
                process_text_file.click(process_text_files, [text_file], text_file_output)

            with gr.Tab("Vector Database"):
                with gr.Row():
                    # gr.Interface
                    vector_database = gr.File(label="Upload Vector Database", file_count="directory")
                    name = gr.Textbox(label="Name")
                vector_db_output = gr.Textbox(label="Vector Database Output")
                process_vector_db = gr.Button("Process Vector Database")
                process_vector_db.click(process_vector_database, [vector_database, name], vector_db_output)

    with gr.Row():
        logs = gr.Textbox(label="Console Output", lines=12, max_lines=12)

    app.load(read_logs, None, logs, every=1)


def run_app():
    app.launch(server_port=7861)
