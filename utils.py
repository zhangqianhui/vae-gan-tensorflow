import os
import errno
import numpy as np
import scipy
import scipy.misc
import tensorflow as tf

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

class CelebA(object):

    def __init__(self):

        self.dataname = "celeba"
        self.dims = 64*64
        self.shape = [64 , 64 , 3]
        self.image_size = 64

    # load celebA dataset
    def load_celebA(self , image_path):

        # get the list of image path
        images_list = read_image_list(image_path)
        # get the data array of image
        return images_list

    @staticmethod
    def getShapeForData(filenames):
        array = [get_image(batch_file, 108, is_crop=True, resize_w = 64,
                           is_grayscale=False) for batch_file in filenames]

        sample_images = np.array(array)
        # return sub_image_mean(array , IMG_CHANNEL)
        return sample_images

    @staticmethod
    def getNextBatch(input_list, maxiter_num,  batch_num, batch_size=128):

        if batch_num >= maxiter_num - 1:

            length = len(input_list)
            perm = np.arange(length)
            np.random.shuffle(perm)
            input_list = input_list[perm]

        return input_list[(batch_num) * batch_size: (batch_num + 1) * batch_size]


def get_image(image_path , image_size , is_crop=True, resize_w = 64 , is_grayscale = False):
    return transform(imread(image_path , is_grayscale), image_size, is_crop , resize_w)

def transform(image, npx = 64 , is_crop=False, resize_w = 64):

    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image , npx , resize_w = resize_w)
    else:
        cropped_image = image
        cropped_image = scipy.misc.imresize(cropped_image ,
                            [resize_w , resize_w])
    return np.array(cropped_image)/127.5 - 1

def center_crop(x, crop_h , crop_w=None, resize_w=64):

    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    # return scipy.misc.imresize(x[40:218-30, 15:178-15],
    #                            [resize_w, resize_w])
    return scipy.misc.imresize(x[j:j + crop_h, i:i + crop_w],
                               [resize_w, resize_w])

# def get_image(image_path, is_grayscale=False):
#     return np.array(inverse_transform(imread(image_path, is_grayscale)))

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path, is_grayscale=False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w: i * w + w, :] = image

    return img

def inverse_transform(image):
    return (image + 1.) / 2.

def read_image_list(category):

    filenames = []
    print("list file")
    list = os.listdir(category)
    for file in list:
        filenames.append(category + "/" + file)
    print("list file ending!")

    length = len(filenames)
    perm = np.arange(length)
    np.random.shuffle(perm)
    filenames = np.array(filenames)
    filenames = filenames[perm]

    return filenames

def sample_label():

    num = 64
    label_vector = np.zeros((num , 128), dtype=np.float)
    for i in range(0 , num):
        label_vector[i , (i/8)%2] = 1.0
    return label_vector

def log10(x):

  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator



