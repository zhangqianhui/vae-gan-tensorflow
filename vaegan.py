import tensorflow as tf
from Distribution import Product , Gaussian , Categorical ,Bernoulli,MeanBernoulli
from ops import batch_normal, lrelu, de_conv, conv2d, fully_connect, instance_norm, residual, deresidual
from utils import save_images
from utils import CelebA
from tensorflow.contrib.layers import flatten
import numpy as np
import cv2
import os

TINY = 1e-8

class vaegan(object):

    #build model
    def __init__(self, batch_size, max_epoch, model_path, data,
                 network_type , sample_size, sample_path, log_dir, gen_learning_rate, dis_learning_rate, info_reg_coeff):

        self.batch_size = batch_size
        self.max_epoch = max_epoch

        self.infogan_model_path = model_path[0]

        self.ds_train = data
        self.type = network_type
        self.sample_size = sample_size
        self.sample_path = sample_path
        self.log_dir = log_dir
        self.learning_rate_gen = gen_learning_rate
        self.learning_rate_dis = dis_learning_rate
        self.info_reg_coeff = info_reg_coeff
        self.log_vars = []

        self.channel = 3

        self.output_size = CelebA().image_size

        self.images = tf.placeholder(tf.float32, [self.batch_size, self.output_size, self.output_size, self.channel])

        self.z_p = tf.placeholder(tf.float32, [self.batch_size , self.sample_size])
        self.ep = tf.random_normal(shape=[self.batch_size, 1024])
        self.y = tf.placeholder(tf.float32, [self.batch_size , 2])

    def build_model_infoGan(self):

        #encode
        self.z_mean, self.z_sigm = self.Encode(self.images)

        self.z_x = tf.add(self.z_mean, tf.sqrt(tf.exp(self.z_sigm))*self.ep)

        self.x_tilde = self.generate(self.z_x, reuse=False)

        #the feature
        self.l_x_tilde, self.De_pro_tilde = self.discriminate(self.x_tilde)

        #for Gan generator
        self.x_p = self.generate(self.z_p, reuse=True)

        # the loss of dis network
        self.l_x,  self.D_pro_logits = self.discriminate(self.images, True)

        _, self.G_pro_logits = self.discriminate(self.x_p, True)
        # the defination of loss

        #KL loss
        self.kl_loss = self.KL_loss(self.z_mean, self.z_sigm)

        #optimize D

        self.D_fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.G_pro_logits), logits=self.G_pro_logits))
        self.D_real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.D_pro_logits), logits=self.D_pro_logits))
        self.D_tilde_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.De_pro_tilde), logits=self.De_pro_tilde))

        #Optimize G

        self.G_fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.G_pro_logits), logits=self.G_pro_logits))
        self.G_tilde_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.De_pro_tilde), logits=self.De_pro_tilde))

        #for Dis
        self.D_loss = self.D_fake_loss + self.D_real_loss + self.D_tilde_loss

        self.LL_loss = tf.reduce_mean(tf.square(self.l_x - self.l_x_tilde))

        #For encode
        self.encode_loss = self.kl_loss/(1024*64) + self.LL_loss

        #for Gen
        self.g_loss = self.G_fake_loss + self.G_tilde_loss + self.LL_loss

        self.log_vars.append(("encode_loss", self.encode_loss))
        self.log_vars.append(("generator_loss", self.g_loss))
        self.log_vars.append(("discriminator_loss", self.D_loss))

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'dis' in var.name]
        self.g_vars = [var for var in t_vars if 'gen' in var.name]
        self.e_vars = [var for var in t_vars if 'e_' in var.name]

        print "d_vars", len(self.d_vars)
        print "g_vars", len(self.g_vars)
        print "e_vars", len(self.e_vars)

        self.saver = tf.train.Saver()
        for k, v in self.log_vars:
            tf.summary.scalar(k, v)

    #do train
    def train(self):

        opti_D = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate_dis).minimize(self.D_loss , var_list=self.d_vars)
        opti_G = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate_gen).minimize(self.g_loss, var_list=self.g_vars)
        opti_e = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate_dis).minimize(self.encode_loss, var_list=self.e_vars)

        init = tf.global_variables_initializer()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            sess.run(init)

            self.saver.restore(sess, self.infogan_model_path)

            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)

            batch_num = 0
            e = 0
            step = 0
            #get the []
            #z_var = self.z_var.eval()
            while e <= self.max_epoch:

                max_iter = len(self.ds_train)/self.batch_size - 1

                while batch_num < len(self.ds_train)/self.batch_size:

                    step = step + 1

                    train_list = CelebA.getNextBatch(self.ds_train, max_iter , batch_num, self.batch_size)
                    realbatch_array = CelebA.getShapeForData(train_list)
                    sample_z = np.random.normal(size=[self.batch_size, self.sample_size])

                    sess.run(opti_e, feed_dict={self.images: realbatch_array})
                    #optimization D
                    sess.run(opti_D, feed_dict={self.images: realbatch_array, self.z_p: sample_z})
                    #optimizaiton G
                    sess.run(opti_G, feed_dict={self.images: realbatch_array, self.z_p: sample_z})

                    summary_str = sess.run(summary_op, feed_dict = {self.images:realbatch_array, self.z_p: sample_z})
                    summary_writer.add_summary(summary_str , step)

                    batch_num += 1

                    if step%20 == 0:

                        D_loss = sess.run(self.D_loss, feed_dict={self.images: realbatch_array, self.z_p: sample_z})
                        fake_loss = sess.run(self.g_loss, feed_dict={self.images: realbatch_array, self.z_p: sample_z})
                        encode_loss = sess.run(self.encode_loss, feed_dict={self.images: realbatch_array, self.z_p: sample_z})
                        print("EPOCH %d step %d: D: loss = %.7f G: loss=%.7f Encode: loss=%.7f" % (e, step, D_loss, fake_loss, encode_loss))

                    if np.mod(step , 200) == 1:

                        save_images(realbatch_array[0:100], [10, 10],
                                    '{}/train_{:02d}_{:04d}_r.png'.format(self.sample_path, e, step))

                    if np.mod(step , 200) == 1:

                        sample_images = sess.run(self.x_tilde, feed_dict={self.images: realbatch_array})
                        save_images(sample_images[0:100] , [10 , 10], '{}/train_{:02d}_{:04d}.png'.format(self.sample_path, e, step))
                        self.saver.save(sess , self.infogan_model_path)

                e += 1
                batch_num = 0

            save_path = self.saver.save(sess , self.infogan_model_path)
            print "Model saved in file: %s" % save_path

    #do test
    def test(self):

        flag = 0

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config= config) as sess:

            sess.run(init)

            self.saver.restore(sess , self.infogan_model_path)

            flag = 18

            for i in range(0, flag):

                train_list = CelebA.getNextBatch(self.ds_train, np.inf, i, self.batch_size)
                realbatch_array = CelebA.getShapeForData(train_list)

                output_image = sess.run(self.x_tilde, feed_dict={self.images: realbatch_array})

                save_images(output_image[0: 0 + 64], [8, 8], './{}/test{:02d}_{:04d}.png'.format(self.sample_path , i, 0))
                save_images(realbatch_array[0: 0 + 64], [8, 8], './{}/test{:02d}_{:04d}_r.png'.format(self.sample_path , i, 0))

            image1 = cv2.imread('./{}/test{:02d}_{:04d}.png'.format(self.sample_path , 0, 0), 1)
            image2 = cv2.imread('./{}/test{:02d}_{:04d}_r.png'.format(self.sample_path , 0, 0), 1)
            cv2.imshow("test", image1)
            cv2.imshow("real_image", image2)

            cv2.waitKey(-1)

            print("Test finish!")

    def test2(self):

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(init)

            self.saver.restore(sess, self.infogan_model_path)
            flag = 18

            for i in range(0, flag):

                train_list = CelebA.getNextBatch(self.ds_train, np.inf, i, self.batch_size)
                realbatch_array = CelebA.getShapeForData(train_list)

                output_image = sess.run(self.x_tilde, feed_dict={self.images: realbatch_array})

                for j in range(0, 64):

                    save_images(output_image[j: j + 1], [1, 1],
                                './{}/test{:02d}_{:04d}.png'.format(self.sample_path, i, j))
                    save_images(realbatch_array[j: j + 1], [1, 1],
                                './{}/test{:02d}_{:04d}_r.png'.format(self.sample_path, i, j))

            image1 = cv2.imread('./{}/test{:02d}_{:04d}.png'.format(self.sample_path, 0, 0), 1)
            image2 = cv2.imread('./{}/test{:02d}_{:04d}_r.png'.format(self.sample_path, 0, 0), 1)
            cv2.imshow("test", image1)
            cv2.imshow("real_image", image2)

            cv2.waitKey(-1)

            print("Test finish!")

    def test3(self):

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            sess.run(init)

            self.saver.restore(sess, self.infogan_model_path)

            train_list = CelebA.getNextBatch(self.ds_train, 14, self.batch_size)
            realbatch_array = CelebA.getShapeForData(train_list)

            output_image = sess.run(self.x_tilde, feed_dict={self.images: realbatch_array})

            flag1 = 0
            flag2 = 64
            flag3 = 8

            save_images(output_image[flag1: flag2], [flag3, flag3],
                        './{}/test{:02d}_{:04d}.png'.format(self.sample_path, 0, 0))
            save_images(realbatch_array[flag1: flag2], [flag3, flag3],
                        './{}/test{:02d}_{:04d}_r.png'.format(self.sample_path, 0, 0))

            image1 = cv2.imread('./{}/test{:02d}_{:04d}.png'.format(self.sample_path, 0, 0), 1)
            image2 = cv2.imread('./{}/test{:02d}_{:04d}_r.png'.format(self.sample_path, 0, 0), 1)
            cv2.imshow("test", image1)
            cv2.imshow("real_image", image2)

            cv2.waitKey(-1)

            print("Test finish!")

    def discriminate(self, x_var, reuse=False):

        with tf.variable_scope("discriminator") as scope:

            if reuse:
                scope.reuse_variables()

            conv1 = lrelu(conv2d(x_var, output_dim=32, name='dis_conv1'))

            conv2= lrelu(batch_normal(conv2d(conv1, output_dim=128, name='dis_conv2'), scope='dis_bn1', reuse=reuse))
            conv3= lrelu(batch_normal(conv2d(conv2, output_dim=256, name='dis_conv3'), scope='dis_bn2', reuse=reuse))
            conv4= lrelu(batch_normal(conv2d(conv3, output_dim=256, name='dis_conv4'), scope='dis_bn3', reuse=reuse))

            conv4 = tf.reshape(conv4, [self.batch_size, -1])

            fl = tf.nn.relu(batch_normal(fully_connect(conv4, output_size=512, scope='dis_fully1'),scope='dis_bn4', reuse=reuse))

            output = fully_connect(conv4, output_size=1, scope='dis_fully2')

            return fl, output

    def generate(self, z_var, reuse=False):

        with tf.variable_scope('generator') as scope:

            if reuse == True:
                scope.reuse_variables()

            d1 = tf.nn.relu(batch_normal(fully_connect(z_var , output_size=8*8*256, scope='gen_fully1'), scope='gen_bn1', reuse=reuse))
            d2 = tf.reshape(d1, [self.batch_size, 8, 8, 256])
            d2 = tf.nn.relu(batch_normal(de_conv(d2 , output_shape=[self.batch_size, 16, 16, 256], name='gen_deconv2'), scope='gen_bn2', reuse=reuse))
            d3 = tf.nn.relu(batch_normal(de_conv(d2, output_shape=[self.batch_size, 32, 32, 128], name='gen_deconv3'), scope='gen_bn3', reuse=reuse))
            d4 = tf.nn.relu(batch_normal(de_conv(d3, output_shape=[self.batch_size, 64, 64, 32], name='gen_deconv4'), scope='gen_bn4', reuse=reuse))
            d5 = de_conv(d4, output_shape=[self.batch_size, 64, 64, 3], name='gen_deconv5', d_h=1, d_w=1)

            return tf.nn.tanh(d5)

    def Encode(self, x):

        with tf.variable_scope('encode') as scope:

            conv1 = tf.nn.relu(batch_normal(conv2d(x, output_dim=64, name='e_c1'), scope='e_bn1'))
            conv2 = tf.nn.relu(batch_normal(conv2d(conv1, output_dim=128, name='e_c2'), scope='e_bn2'))
            conv3 = tf.nn.relu(batch_normal(conv2d(conv2 , output_dim=256, name='e_c3'), scope='e_bn3'))
            conv3 = tf.reshape(conv3, [self.batch_size, 256 * 8 * 8])

            z_mean = tf.nn.relu(batch_normal(fully_connect(conv3 , output_size=1024, scope='e_f4'), scope='e_bn4'))
            z_sigma = tf.nn.relu(batch_normal(fully_connect(conv3, output_size=1024, scope='e_f5'), scope='e_bn5'))

            return z_mean, z_sigma

    def KL_loss(self, mu, log_var):
        return -0.5 * tf.reduce_sum(1 + log_var - tf.pow(mu, 2) - tf.exp(log_var))

    def sample_z(self, mu, log_var):
        eps = tf.random_normal(shape=tf.shape(mu))
        return mu + tf.exp(log_var / 2) * eps







