import os
from typing import List
import pathlib


def find_folders_with_no_audio_files(root_data_dir: str):
    num_folders = 0
    folder_stats = {}
    
    cwd = os.getcwd()
    os.chdir(root_data_dir)
    for foldername in os.listdir(root_data_dir):
        num_folders += 1
        all_audio_file_paths_in_target_dir: List[str] = []
        for file in pathlib.Path(os.path.abspath(foldername)).rglob('*.wav'):
            all_audio_file_paths_in_target_dir.append(os.path.abspath(file))
        num_audio_file_paths_in_target_dir = len(all_audio_file_paths_in_target_dir)
        assert num_audio_file_paths_in_target_dir != 0, os.path.abspath(foldername)
        folder_stats[foldername] = {
            'Num_Audio_Files': num_audio_file_paths_in_target_dir
        }
    os.chdir(cwd)


if __name__ == '__main__':
    find_folders_with_no_audio_files(root_data_dir='D:\\data\\Bees\\beemon\\raw\\rpi4-2')
