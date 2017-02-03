import numpy as np
import os, sys, getopt

# Main path to your caffe installation
#todo change to your caffe home location
caffe_root = '/home/magedmilad/caffe/'
# needed to be adjusted on every device 

# Model prototxt file
model_prototxt = caffe_root + 'models/bvlc_googlenet/deploy.prototxt'

# Model caffemodel file
model_trained = caffe_root + 'models/bvlc_googlenet/bvlc_googlenet.caffemodel'

# File containing the class labels
imagenet_labels = caffe_root + 'data/ilsvrc12/synset_words.txt'

# Path to the mean image (used for input processing)
mean_path = caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'

# Name of the layer we want to extract
layer_name = 'pool5/7x7_s1'

sys.path.insert(0, caffe_root + 'python')
import caffe

def main(argv):
    folder = argv[1]

    # Setting this to CPU, but feel free to use GPU if you have CUDA installed
    caffe.set_mode_cpu()
    # Loading the Caffe model, setting preprocessing parameters
    net = caffe.Classifier(model_prototxt, model_trained,
                           mean=np.load(mean_path).mean(1).mean(1),
                           channel_swap=(2,1,0),
                           raw_scale=255,
                           image_dims=(256, 256))

    # Loading class labels
    with open(imagenet_labels) as f:
        labels = f.readlines()


    writer = open(folder+"out.txt","w");
    for image_path in os.listdir(folder):
        input_image = caffe.io.load_image(image_path)
        prediction = net.predict([input_image], oversample=False)
        np.savetxt(writer, net.blobs[layer_name].data[0].reshape(1, -1), fmt='%.8g')
        writer.write('\n')
        writer.truncate()
    writer.close()

if __name__ == "__main__":
    main(sys.argv)
