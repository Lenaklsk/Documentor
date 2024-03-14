import os
from pytube import Playlist, YouTube
import ffmpeg
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import TEMP_FILE_STORAGE_DIR, MP4S_DIR, MP3S_DIR


YT_BASE_URL = "https://www.youtube.com/watch?v="
URL_FILE_PATH = os.path.join(TEMP_FILE_STORAGE_DIR, "video_urls.txt")
AUDIO_BASE_PATH = MP4S_DIR

def check_yt_url_type_and_process(input_urls):
    playlist_pattern = "&list="
    print(input_urls)
    print(len(input_urls))
    if len(input_urls) == 1 and playlist_pattern in input_urls[0]:
        get_urls_file_from_playlist(input_urls[0])
    elif len(input_urls) > 1 and any(playlist_pattern in url for url in input_urls):
        print("Error: Can't process multiple yt playlists at once")
        return
    else:
        get_urls_file_from_yt_video_links(input_urls)
            
def get_urls_file_from_playlist(playlist_url):
    print("Playlist:", playlist_url)
    playlist = Playlist(playlist_url)
    video_urls = [v[32:] for v in playlist]
    print(video_urls)
    create_urls_text_file(video_urls)


def get_urls_file_from_yt_video_links(video_urls):
    print("Videos:", video_urls)
    video_urls = [v[32:] for v in video_urls]
    print(video_urls)
    create_urls_text_file(video_urls)


def create_urls_text_file(video_urls):
    with open(os.path.join(TEMP_FILE_STORAGE_DIR, "video_urls.txt"), "w") as urls_txt:
        urls_txt.write("\n".join(video_urls))


# TODO Stimmt hier mp4s oder soll das mp3s sein?

def download_urls(folder_name):
    while True:
        with open(URL_FILE_PATH, 'r') as file:
            lines = file.readlines()

        if not lines:
            print("Finished downloading all videos!")
            return

        url = lines.pop()
        print(f"URL read: {url}.")
        yt_video = YouTube(YT_BASE_URL + url)
        audio = yt_video.streams.get_audio_only()
        print("Starting download.")
        audio.download(str(os.path.join(AUDIO_BASE_PATH, folder_name)))
        print("Finished download.")

        with open(URL_FILE_PATH, 'w') as file:
            file.writelines(lines)

def convert_to_mp3(archive_name):
    src_dir = str(os.path.join(MP4S_DIR, archive_name))
    dest_dir = str(os.path.join(MP3S_DIR, archive_name))

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for file_name in os.listdir(src_dir):
        if not file_name.lower().endswith(('.mp3', '.wav', '.mp4')):
            continue

        source_path = os.path.join(src_dir, file_name)
        destination_path = os.path.join(dest_dir, os.path.splitext(file_name)[0] + '.mp3')

        if os.path.exists(destination_path):
            continue

        try:
            stream = ffmpeg.input(source_path)
            stream = ffmpeg.output(stream, destination_path)
            ffmpeg.run(stream)
            print(f"Converted {file_name} to MP3.")
        except ffmpeg.Error as e:
            print(f"An error occurred while converting {file_name}: {e}")