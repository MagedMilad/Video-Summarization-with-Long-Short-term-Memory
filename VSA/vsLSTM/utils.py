import numpy as np
from moviepy.editor import VideoFileClip, concatenate_videoclips
import os
import sys
import scipy.io as sio
import os.path as osp

# settings:
#     c - canonical
#     a - augumented
#     t - transfer
# dataset:
#     t - tvsum
#     s - summe

def get_model_name(dataset, setting):
   if(dataset == 't'):
       dataset = 'TVSum'
   else:
       dataset = 'SumMe'
   if(setting == 'c'):
       setting = 'Canonical'
   elif(setting == 'a'):
       setting = 'Augmented'
   else:
       setting = 'Transfer'
   print 'dataset: {}, Setting: {}'.format(dataset,setting)
   return 'dataset: {}, Setting: {}'.format(dataset,setting)

def get_y(vid):
    # print (vid)
    mat_contents = sio.loadmat('{}/{}.mat'.format('/home/magedmilad/Video-Summarization-with-Long-Short-term-Memory/Video Sampling/mat_', vid))
    return mat_contents['gt_score']


def expand_y(out, length, fps):
    vid_frames = np.zeros(int(length))
    mul = fps/2
    last = 0
    for index, val in enumerate(out):
        curr = min(int(index * mul), int(length - 1))
        prev = min(int(last * mul), int(length - 1))
        vid_frames[curr] = val
        if index != 0:
            dy = vid_frames[curr] - vid_frames[prev]
            dx = curr - prev
            if dx == 0:
                for j in range(prev, curr):
                    vid_frames[j] = vid_frames[curr]
            else:
                slope = dy / dx
                for j in range(prev, curr):
                    vid_frames[j] = int(round(vid_frames[curr] - slope * (curr - j)))
        last = index
    for j in range(int(last * mul), int(length)):
        vid_frames[j] = vid_frames[int(last * mul)]
    return vid_frames

def exp_y_mag(out, length, fps):
    length = int(length)
    y = np.zeros(int(length))
    offset = int(fps/2)
    max_i=0
    for min_i, val in enumerate(out):
        if(min_i == 0):
            y[max_i]=out[min_i]
            max_i+=offset
        else:
            if (out[min_i] == out[min_i-1]):
                for i in range(offset):
                    y[min(max_i-i-1,length-1)] = out[min_i]
                y[min(max_i,length-1)] = out[min_i]
                max_i+=offset
            else:
                for i in range(offset/2):
                    y[min(max_i-i-1,length-1)] = out[min_i]
                    y[min(max_i-offset+i,length-1)] = out[min_i-1]
                y[min(max_i,length-1)] = out[min_i]
                max_i += offset
    return y


def frame_to_interval(frames, fps):
    intervals = []
    index = 0
    while index < len(frames):
        if frames[index] == 1:
            start = index
            while index < len(frames) and frames[index] == 1:
                index += 1
            intervals.append((start/fps, (index-1)/fps))
        index += 1
    return intervals


def get_summery(video_name, y):
    y = np.array(y)
    video = VideoFileClip(video_name)
    fps = video.fps
    length = video.duration * fps
    y = exp_y_mag(y, length, fps)
    y = np.array(y)
    return y


def save_video(video_name, y,dest_dir):
    video = VideoFileClip(video_name)
    fps = video.fps
    dir = osp.dirname(video_name)
    name = osp.basename(video_name)
    # length = int(video.fps * video.duration)
    # expanded_y = expand_y(y, length, fps)
    final_vid = concatenate_videoclips([video.subclip(start, end) for (start, end) in frame_to_interval(y, fps)])
    final_vid.to_videofile(dest_dir, fps=fps)

