import tensorflow as tf
from ops import batch_normal, de_conv, conv2d, fully_connect, lrelu
from utils import save_images, get_image
from utils import CelebA
import numpy as np
import cv2 
from tensorflow.python.framework.ops import convert_to_tensor
import os
TINY = 1e-8
d_scale_factor = 0.25
g_scale_factor =  1 - 0.75/2


class vaegan(object):

    #build model
    def __init__(self, batch_size, max_iters, repeat, model_path, data_ob, latent_dim, sample_path, log_dir, learnrate_init):

        self.batch_size = batch_size
        self.max_iters = max_iters
        self.repeat_num = repeat
        self.saved_model_path = model_path
        self.data_ob = data_ob
        self.latent_dim = latent_dim
        self.sample_path = sample_path
        self.log_dir = log_dir
        self.learn_rate_init = learnrate_init
        self.log_vars = []

        self.channel = 3
        self.output_size = data_ob.image_size
        self.images = tf.placeholder(tf.float32, [self.batch_size, self.output_size, self.output_size, self.channel])
        self.ep = tf.random_normal(shape=[self.batch_size, self.latent_dim])
        self.zp = tf.random_normal(shape=[self.batch_size, self.latent_dim])

        self.dataset = tf.data.Dataset.from_tensor_slices(
            convert_to_tensor(self.data_ob.train_data_list, dtype=tf.string))
        self.dataset = self.dataset.map(lambda filename : tuple(tf.py_func(self._read_by_function,
                                                                            [filename], [tf.double])), num_parallel_calls=16)
        self.dataset = self.dataset.repeat(10000)
        self.dataset = self.dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))

        self.iterator = tf.data.Iterator.from_structure(self.dataset.output_types, self.dataset.output_shapes)
        self.next_x = tf.squeeze(self.iterator.get_next())
        self.training_init_op = self.iterator.make_initializer(self.dataset)

    def build_model_vaegan(self):

        self.z_mean, self.z_sigm = self.Encode(self.images)
        self.z_x = tf.add(self.z_mean, tf.sqrt(tf.exp(self.z_sigm))*self.ep)
        self.x_tilde = self.generate(self.z_x, reuse=False)
        self.l_x_tilde, self.De_pro_tilde = self.discriminate(self.x_tilde)

        self.x_p = self.generate(self.zp, reuse=True)

        self.l_x,  self.D_pro_logits = self.discriminate(self.images, True)
        _, self.G_pro_logits = self.discriminate(self.x_p, True)

        #KL loss
        self.kl_loss = self.KL_loss(self.z_mean, self.z_sigm)

        # D loss
        self.D_fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.G_pro_logits), logits=self.G_pro_logits))
        self.D_real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.D_pro_logits) - d_scale_factor, logits=self.D_pro_logits))
        self.D_tilde_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.De_pro_tilde), logits=self.De_pro_tilde))

        # G loss
        self.G_fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.G_pro_logits) - g_scale_factor, logits=self.G_pro_logits))
        self.G_tilde_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.De_pro_tilde) - g_scale_factor, logits=self.De_pro_tilde))

        self.D_loss = self.D_fake_loss + self.D_real_loss + self.D_tilde_loss

        # preceptual loss(feature loss)
        self.LL_loss = tf.reduce_mean(tf.reduce_sum(self.NLLNormal(self.l_x_tilde, self.l_x), [1,2,3]))

        #For encode
        self.encode_loss = self.kl_loss/(self.latent_dim*self.batch_size) - self.LL_loss / (4 * 4 * 256)

        #for Gen
        self.G_loss = self.G_fake_loss + self.G_tilde_loss - 1e-6*self.LL_loss

        self.log_vars.append(("encode_loss", self.encode_loss))
        self.log_vars.append(("generator_loss", self.G_loss))
        self.log_vars.append(("discriminator_loss", self.D_loss))
        self.log_vars.append(("LL_loss", self.LL_loss))

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'dis' in var.name]
        self.g_vars = [var for var in t_vars if 'gen' in var.name]
        self.e_vars = [var for var in t_vars if 'e_' in var.name]

        self.saver = tf.train.Saver()
        for k, v in self.log_vars:
            tf.summary.scalar(k, v)

    #do train
    def train(self):

        global_step = tf.Variable(0, trainable=False)
        add_global = global_step.assign_add(1)
        new_learning_rate = tf.train.exponential_decay(self.learn_rate_init, global_step=global_step, decay_steps=10000,
                                                   decay_rate=0.98)
        #for D
        trainer_D = tf.train.RMSPropOptimizer(learning_rate=new_learning_rate)
        gradients_D = trainer_D.compute_gradients(self.D_loss, var_list=self.d_vars)
        opti_D = trainer_D.apply_gradients(gradients_D)

        #for G
        trainer_G = tf.train.RMSPropOptimizer(learning_rate=new_learning_rate)
        gradients_G = trainer_G.compute_gradients(self.G_loss, var_list=self.g_vars)
        opti_G = trainer_G.apply_gradients(gradients_G)

        #for E
        trainer_E = tf.train.RMSPropOptimizer(learning_rate=new_learning_rate)
        gradients_E = trainer_E.compute_gradients(self.encode_loss, var_list=self.e_vars)
        opti_E = trainer_E.apply_gradients(gradients_E)

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            sess.run(init)

            # Initialzie the iterator
            sess.run(self.training_init_op)
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)

            #self.saver.restore(sess, self.saved_model_path)
            step = 0

            while step <= self.max_iters:

                next_x_images = sess.run(self.next_x)

                fd ={self.images: next_x_images}
                sess.run(opti_E, feed_dict=fd)
                # optimizaiton G
                sess.run(opti_G, feed_dict=fd)
                # optimization D
                sess.run(opti_D, feed_dict=fd)

                summary_str = sess.run(summary_op, feed_dict=fd)

                summary_writer.add_summary(summary_str, step)
                new_learn_rate = sess.run(new_learning_rate)

                if new_learn_rate > 0.00005:
                    sess.run(add_global)

                if step%200 == 0:

                    D_loss, fake_loss, encode_loss, LL_loss, kl_loss, new_learn_rate \
                        = sess.run([self.D_loss, self.G_loss, self.encode_loss, self.LL_loss, self.kl_loss/(128*64), new_learning_rate], feed_dict=fd)
                    print("Step %d: D: loss = %.7f G: loss=%.7f E: loss=%.7f LL loss=%.7f KL=%.7f, LR=%.7f" % (step, D_loss, fake_loss, encode_loss, LL_loss, kl_loss, new_learn_rate))

                if np.mod(step , 200) == 1:

                    save_images(next_x_images[0:64], [8, 8],
                                '{}/train_{:02d}_real.png'.format(self.sample_path, step))
                    sample_images = sess.run(self.x_tilde, feed_dict=fd)
                    save_images(sample_images[0:64] , [8 , 8], '{}/train_{:02d}_recon.png'.format(self.sample_path, step))

                if np.mod(step , 2000) == 1 and step != 0:

                    self.saver.save(sess , self.saved_model_path)

                step += 1

            save_path = self.saver.save(sess , self.saved_model_path)
            print "Model saved in file: %s" % save_path

    def test(self):

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            # Initialzie the iterator
            sess.run(self.training_init_op)

            sess.run(init)
            self.saver.restore(sess, self.saved_model_path)

            next_x_images = sess.run(self.next_x)

            real_images, sample_images = sess.run([self.images, self.x_tilde], feed_dict={self.images: next_x_images})
            save_images(sample_images[0:64], [8, 8], '{}/train_{:02d}_{:04d}_con.png'.format(self.sample_path, 0, 0))
            save_images(real_images[0:64], [8, 8], '{}/train_{:02d}_{:04d}_r.png'.format(self.sample_path, 0, 0))

            ri = cv2.imread('{}/train_{:02d}_{:04d}_r.png'.format(self.sample_path, 0, 0), 1)
            fi = cv2.imread('{}/train_{:02d}_{:04d}_con.png'.format(self.sample_path, 0, 0), 1)

            cv2.imshow('real_image', ri)
            cv2.imshow('reconstruction', fi)

            cv2.waitKey(-1)

    def discriminate(self, x_var, reuse=False):

        with tf.variable_scope("discriminator") as scope:

            if reuse:
                scope.reuse_variables()

            conv1 = tf.nn.relu(conv2d(x_var, output_dim=32, name='dis_conv1'))
            conv2= tf.nn.relu(batch_normal(conv2d(conv1, output_dim=128, name='dis_conv2'), scope='dis_bn1', reuse=reuse))
            conv3= tf.nn.relu(batch_normal(conv2d(conv2, output_dim=256, name='dis_conv3'), scope='dis_bn2', reuse=reuse))
            conv4 = conv2d(conv3, output_dim=256, name='dis_conv4')
            middle_conv = conv4
            conv4= tf.nn.relu(batch_normal(conv4, scope='dis_bn3', reuse=reuse))
            conv4= tf.reshape(conv4, [self.batch_size, -1])

            fl = tf.nn.relu(batch_normal(fully_connect(conv4, output_size=256, scope='dis_fully1'), scope='dis_bn4', reuse=reuse))
            output = fully_connect(fl , output_size=1, scope='dis_fully2')

            return middle_conv, output

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
            fc1 = tf.nn.relu(batch_normal(fully_connect(conv3, output_size=1024, scope='e_f1'), scope='e_bn4'))
            z_mean = fully_connect(fc1 , output_size=128, scope='e_f2')
            z_sigma = fully_connect(fc1, output_size=128, scope='e_f3')

            return z_mean, z_sigma

    def KL_loss(self, mu, log_var):
        return -0.5 * tf.reduce_sum(1 + log_var - tf.pow(mu, 2) - tf.exp(log_var))

    def sample_z(self, mu, log_var):
        eps = tf.random_normal(shape=tf.shape(mu))
        return mu + tf.exp(log_var / 2) * eps

    def NLLNormal(self, pred, target):

        c = -0.5 * tf.log(2 * np.pi)
        multiplier = 1.0 / (2.0 * 1)
        tmp = tf.square(pred - target)
        tmp *= -multiplier
        tmp += c

        return tmp

    def _parse_function(self, images_filenames):

        image_string = tf.read_file(images_filenames)
        image_decoded = tf.image.decode_and_crop_jpeg(image_string, crop_window=[218 / 2 - 54, 178 / 2 - 54 , 108, 108], channels=3)
        image_resized = tf.image.resize_images(image_decoded, [self.output_size, self.output_size])
        image_resized = image_resized / 127.5 - 1

        return image_resized

    def _read_by_function(self, filename):

        array = get_image(filename, 108, is_crop=True, resize_w=self.output_size,
                           is_grayscale=False)
        real_images = np.array(array)
        return real_images











