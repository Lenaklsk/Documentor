import gc
import os

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

cwd = os.path.abspath(os.path.dirname(__file__))

from config import MP3S_DIR, TEMP_FILE_STORAGE_DIR, DOCUMENT_COLLECTIONS_DIR

MP3_FILE_NAMES = "mp3_file_names.txt"


def get_list_of_files_to_transcribe(folder_name):
    file_names = os.listdir(os.path.join(MP3S_DIR, folder_name))

    path_to_output_file = os.path.join(TEMP_FILE_STORAGE_DIR, "mp3_file_names.txt")
    print(path_to_output_file)
    with open(path_to_output_file, 'w+', encoding="utf-8") as file:
        file.writelines("\n".join(file_names))


def setup_model_for_transcription(folder_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    speech_recognition_pipeline = pipeline(
        "automatic-speech-recognition",
        generate_kwargs={"task": "translate"},
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )

    transcribe_batch(folder_name, speech_recognition_pipeline)

    # Delete model and processor to free up GPU memory
    del speech_recognition_pipeline
    del processor
    del model

    # Clearing the cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Invoking garbage collection explicitly
    gc.collect()


# TODO On better hardware it might be better to not do one after the other because of the batch_size option?
# def transcribe_batch(file_paths_txt, audio_dir, output_dir):
def transcribe_batch(folder_name, speech_recognition_pipeline):
    path_to_mp3_file_names = os.path.join(TEMP_FILE_STORAGE_DIR, MP3_FILE_NAMES)
    while True:
        with open(path_to_mp3_file_names, 'r', encoding="utf-8") as file:
            lines_nl = file.readlines()

        if not lines_nl:
            print("Finished transcribing all videos!")
            return

        lines = [line.rstrip('\n') for line in lines_nl]
        input_file_name = lines.pop()
        mp3_path = os.path.join(MP3S_DIR, folder_name, input_file_name)

        print(f"Transcribing file: {input_file_name}.")

        transcription = speech_recognition_pipeline(mp3_path)
        transcription_text_by_chunk = "\n".join([c["text"] for c in transcription["chunks"]])

        os.makedirs(os.path.join(DOCUMENT_COLLECTIONS_DIR, folder_name), exist_ok=True)

        with open(os.path.join(DOCUMENT_COLLECTIONS_DIR, folder_name,
                               (os.path.splitext(input_file_name)[0] + ".txt")), "w+",
                  encoding="utf-8") as output_file:
            output_file.write(transcription_text_by_chunk)

        print(f"Finished transcribing: {input_file_name}.")

        with open(path_to_mp3_file_names, 'w',
                  encoding="utf-8") as file:
            file.writelines('\n'.join(lines))
