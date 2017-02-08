import imageio
import cv2
import os
import sys


VID_NAME = sys.argv[1]
FOLDER_NAME = sys.argv[2]
#VID_NAME = 'test.mp4'
#FOLDER_NAME = VID_NAME.split('.')[0]
FRAMES = 2
SIZE = (256, 256)

vid = imageio.get_reader(VID_NAME,  'ffmpeg')
num_frames = vid.get_meta_data()['nframes']
fps = int(vid.get_meta_data()['fps'])

VID_NAME = sys.argv[1].split('/')
VID_NAME = VID_NAME[len(VID_NAME) - 1]
VID_NAME = VID_NAME.split('.')[0]

path = "./{}/{}/".format(FOLDER_NAME, VID_NAME)
if not os.path.isdir(path):
	os.makedirs(path)

for i in range(0, num_frames-1, fps):
	cv2.imwrite("{}image{}.jpg".format(path, i), cv2.resize(vid.get_data(i), SIZE))
	cv2.imwrite("{}image{}.jpg".format(path, i + int(fps / 2)), cv2.resize(vid.get_data(min(i + int(fps / 2), num_frames-1)), SIZE))