import os
import errno
import numpy as np
import scipy
import scipy.misc


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale=False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)


def transform(image, npx=64, is_crop=False, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
        cropped_image = scipy.misc.imresize(cropped_image,
                                            [resize_w, resize_w])
    return np.array(cropped_image) / 127.5 - 1

def center_crop(x, crop_h , crop_w=None, resize_w=64):

    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
                               [resize_w, resize_w])


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
    return ((image + 1) * 127.5).astype('int32')

class CelebA(object):
    def __init__(self, images_path):

        self.dataname = "CelebA"
        self.dims = 64 * 64
        self.shape = [64, 64, 3]
        self.image_size = 64
        self.channel = 3
        self.images_path = images_path
        self.train_data_list, self.train_lab_list = self.load_celebA()

    def load_celebA(self):

        # get the list of image path
        return read_image_list_file(self.images_path, is_test=False)

    def load_test_celebA(self):

        # get the list of image path
        return read_image_list_file(self.images_path, is_test=True)

def read_image_list_file(category, is_test):
    end_num = 0
    if is_test == False:

        start_num = 1202
        path = category + "celebA/"

    else:

        start_num = 4
        path = category + "celeba_test/"
        end_num = 1202

    list_image = []
    list_label = []

    lines = open(category + "list_attr_celeba.txt")
    li_num = 0
    for line in lines:

        if li_num < start_num:
            li_num += 1
            continue

        if li_num >= end_num and is_test == True:
            break

        flag = line.split('1 ', 41)[20]  # get the label for gender
        file_name = line.split(' ', 1)[0]

        # print flag
        if flag == ' ':

            list_label.append(1)

        else:

            list_label.append(0)

        list_image.append(path + file_name)

        li_num += 1

    lines.close()

    return list_image, list_label



