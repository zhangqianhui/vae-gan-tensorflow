import tensorflow as tf
import os
import string
from utils import log10


def read_image_list(category):
    file_ori = []
    file_new = []

    list = os.listdir(category)
    for file in list:

        if string.find(file, 'r') != -1:
            file_ori.append(category + "/" + file)
        else:
            file_new.append(category + "/" + file)

    return file_ori, file_new


def compare(x, y):
    stat_x = os.stat(x)
    stat_y = os.stat(y)

    if stat_x.st_ctime < stat_y.st_ctime:

        return -1

    elif stat_x.st_ctime > stat_y.st_ctime:

        return 1

    else:

        return 0


score = 0.0

file_path = "./vaeganCeleba2/sample_psnr/"
ori_list, gen_list = read_image_list(file_path)
ori_list.sort(compare)
gen_list.sort(compare)

print len(ori_list)

for i in range(len(ori_list)):
    with tf.gfile.FastGFile(ori_list[i]) as image_file:
        img1_str = image_file.read()
    with tf.gfile.FastGFile(gen_list[i]) as image_file:
        img2_str = image_file.read()

    input_img = tf.placeholder(tf.string)
    decoded_image = tf.expand_dims(tf.image.decode_png(input_img, channels=3), 0)

    with tf.Session() as sess:
        img1 = sess.run(decoded_image, feed_dict={input_img: img1_str})
        img2 = sess.run(decoded_image, feed_dict={input_img: img2_str})

        mse_loss = tf.reduce_mean(tf.cast((img1 - img2) ** 2, tf.float32))

        print "mse_loss", sess.run(mse_loss)
        print "psnr", sess.run(20 * log10(255 / tf.sqrt(mse_loss)))
        score += sess.run(20 * log10(255 / tf.sqrt(mse_loss)))


print "result",score / len(ori_list)