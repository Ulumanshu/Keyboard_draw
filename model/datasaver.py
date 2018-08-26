#otenim/Python-EMNIST-Decoder
#modified otenim script for emnist dataset extraction to dirs
#to get example functioning you must download emnist dataset
# in static/Emnist_dir (easy way) or create your own
import os
import bitstring
import numpy as np
from PIL import Image, ImageEnhance


def main():

    img_file = './emnist-balanced-test-images-idx3-ubyte'
    label_file = './emnist-balanced-test-labels-idx1-ubyte'
    label_map_file = './emnist-balanced-mapping.txt'
    target_dir = '/home/wooden/Desktop/Pycharm_projects/Flask_Keras/static/Emnist_dir/balanced/test'
    brightness = 1.0
    sharpness = 2.0
    contrast = 3.0
    images_binfile = os.path.expanduser(img_file)
    print(images_binfile)
    labels_binfile = os.path.expanduser(label_file)
    print(labels_binfile)
    label_mapfile = os.path.expanduser(label_map_file)
    print(label_mapfile)
    output_dir = os.path.expanduser(target_dir)
    print(output_dir)

    # create output root directory (if necessary)
    if os.path.exists(output_dir) == False:
        os.makedirs(output_dir)

    # create class-char-map
    with open(label_mapfile, 'r') as f:
        lines = f.readlines()
        labelmap = {}
        for line in lines:
            class_id = int(line.split(' ')[0])
            char_id = int(line.split(' ')[1])
            labelmap[class_id] = char_id

    # read images binfile header
    images_bitstream = bitstring.ConstBitStream(filename=images_binfile)
    images_bitstream.read('int:32') # magic
    n_images = images_bitstream.read('int:32')
    print(n_images)
    img_width = images_bitstream.read('int:32')
    img_height = images_bitstream.read('int:32')

    # read labels binfile header
    labels_bitstream = bitstring.ConstBitStream(filename=labels_binfile)
    labels_bitstream.read('int:32') # magic
    n_labels = labels_bitstream.read('int:32')
    print(n_labels)

    # validation
    assert n_images == n_labels, 'the number of images is not the same as that of images.'
    n_samples = n_images

    cnt = 0
    for i in range(n_samples):
        cnt += 1

        # read a single label record
        record_label = labels_bitstream.read('uint:8')
        # reconstruct the label id
        label = np.uint8(record_label)
        # decoded label character
        character = labelmap[label]

        # create subdirectory (if necessary)
        subdir = os.path.join(output_dir, str(character))
        if os.path.exists(subdir) == False:
            os.makedirs(subdir)

        # read a single image record
        record_image = images_bitstream.readlist('%d*uint:8' % (img_width*img_height))
        # reconstruct the image data
        pixel_data = np.array(record_image, dtype=np.uint8).reshape(img_height, img_width)
        pixel_data = pixel_data.T
        image = Image.fromarray(pixel_data)

        # apply enhancements
        # brightness
        if brightness != 1.0:
            image = ImageEnhance.Brightness(image).enhance(brightness)
        # sharpness
        if sharpness != 1.0:
            image = ImageEnhance.Sharpness(image).enhance(sharpness)
        # contrast
        if contrast != 1.0:
            image = ImageEnhance.Contrast(image).enhance(contrast)

        # save image
        fname = os.path.join(subdir, '%d.png' % cnt)
        image.save(fname)
        print('(%d/%d) decoded image was saved as %s' % (i+1, n_samples, fname))

if __name__ == '__main__':
    main()
