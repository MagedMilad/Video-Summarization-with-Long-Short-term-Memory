import imageio
import cv2
import os
import threading

def work(file):
    global SIZE

    SIZE = (256, 256)
    frames_path = file.split('.')[0]+'/'
    vid = imageio.get_reader(file, 'ffmpeg')
    num_frames = vid.get_meta_data()['nframes']
    fps = int(vid.get_meta_data()['fps'])
    if not os.path.isdir(frames_path):
        os.makedirs(frames_path)
    arr = []

    for i in range(0, num_frames-1, fps):
        t = threading.Thread(target=save_image, args=('{}image{}.jpg'.format(frames_path, i), vid.get_data(i)))
        t.daemon = True
        t.start()
        arr.append(t)
        t = threading.Thread(target=save_image, args=('{}image{}.jpg'.format(frames_path, i + int(fps / 2)), vid.get_data(min(i + int(fps / 2), num_frames-1))))
        t.daemon = True
        t.start()
        arr.append(t)
    for i in arr:
        i.join()

    return frames_path


def save_image(name, data):
    cv2.imwrite(name, cv2.resize(data, SIZE))
