from InceptionV1.examples.imagenet.classify import feature_extract
from vsLSTM.vsLSTM import predect
from vsLSTM.utils import save_video
from sampling import work
from sampler import save_frames
import argparse, textwrap
from argparse import RawTextHelpFormatter

import shutil


dataset = "t"
setting = "t"
# video_path = "/home/magedmilad/PycharmProjects/VSA/Jumps.webm"



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='vsLSTM', formatter_class=RawTextHelpFormatter)
    parser.add_argument('-p', '--video_path', dest='video_path')
    parser.add_argument('-id', '--id', dest='video_id')
    parser.add_argument('-dir', '--sum_dir', dest='dest_dir')

    args = parser.parse_args()




    src = args.video_path.split('?')[0]
    image_dir = save_frames(src)
    features = feature_extract(image_dir)
    Y = predect(dataset, setting, features, src)

    dest_dir = args.dest_dir + args.video_id + ".mp4"
    save_video(src, Y,dest_dir)
    shutil.rmtree(image_dir)
