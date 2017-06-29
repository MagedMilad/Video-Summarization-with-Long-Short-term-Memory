import os


def save_frames(file_path):
    frames_path = file_path.split('.')[0] + '/'
    if not os.path.isdir(frames_path):
        os.makedirs(frames_path)
    os.system("ffmpeg -i '{}' -s 256x256 -r 2 '{}image%d.jpg'".format(file_path, frames_path))
    return frames_path
