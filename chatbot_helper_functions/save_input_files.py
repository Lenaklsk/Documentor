import os
import sys
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import MP4S_DIR, MP3S_DIR, DOCUMENT_COLLECTIONS_DIR, VECTOR_STORES_DIR

def save_mp4_files(mp4_files, archive_name):
    dest_dir = str(os.path.join(MP4S_DIR, archive_name))
    
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for file_path in mp4_files:  
        file_name = os.path.basename(file_path)  # Extract the filename
        destination_path = os.path.join(dest_dir, file_name)

        shutil.copy(file_path, destination_path)
            
def save_mp3_files(mp3_files, archive_name):
    dest_dir = str(os.path.join(MP3S_DIR, archive_name))

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for file_path in mp3_files:  
        file_name = os.path.basename(file_path)  # Extract the filename
        destination_path = os.path.join(dest_dir, file_name)

        shutil.copy(file_path, destination_path)
            
            
def save_text_files(text_files, archive_name):
    dest_dir = str(os.path.join(DOCUMENT_COLLECTIONS_DIR, archive_name))

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        
    for file_path in text_files:  
        file_name = os.path.basename(file_path)  # Extract the filename
        destination_path = os.path.join(dest_dir, file_name)

        shutil.copy(file_path, destination_path)
        
def save_vector_db(vector_db_files, archive_name):
    dest_dir = str(os.path.join(VECTOR_STORES_DIR, archive_name))

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        
    for file_path in vector_db_files:  
        file_name = os.path.basename(file_path)  # Extract the filename
        destination_path = os.path.join(dest_dir, file_name)

        shutil.copy(file_path, destination_path)