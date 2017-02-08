import imageio
import cv2
import os
import sys
import scipy.io as sio

VID_NAME = sys.argv[1]
FRAMES_FOLDER_NAME = sys.argv[2]
MAT_FOLDER_NAME = sys.argv[3]
OUTPUT_FOLDER_NAME = sys.argv[4]
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

frames_path = './{}/{}/'.format(FRAMES_FOLDER_NAME, VID_NAME)
if not os.path.isdir(frames_path):
	os.makedirs(frames_path)

mat_path = './{}/'.format(MAT_FOLDER_NAME)
if not os.path.isdir(mat_path):
	raise Exception('mat folder not found')

out_path = './{}/'.format(OUTPUT_FOLDER_NAME)
if not os.path.isdir(out_path):
	os.makedirs(out_path)

file=open('{}/{}'.format(out_path, VID_NAME), 'w+')

mat_contents = sio.loadmat('{}/{}.mat'.format(mat_path, VID_NAME))
trainY = mat_contents['gt_score'].T

for i in range(0, num_frames-1, fps):
	cv2.imwrite('{}image{}.jpg'.format(frames_path, i), cv2.resize(vid.get_data(i), SIZE))
	cv2.imwrite('{}image{}.jpg'.format(frames_path, i + int(fps / 2)), cv2.resize(vid.get_data(min(i + int(fps / 2), num_frames-1)), SIZE))
	file.write('{} {}'.format(trainY[0][i], trainY[0][min(i + int(fps / 2), num_frames-1)] ))
	if i+fps < num_frames-1:
		file.write(' ')