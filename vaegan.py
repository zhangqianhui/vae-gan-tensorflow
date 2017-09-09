import tensorflow as tf
from ops import batch_normal, de_conv, conv2d, fully_connect, lrelu
from utils import save_images
from utils import CelebA
import numpy as np

TINY = 1e-8

class vaegan(object):

    #build model
    def __init__(self, batch_size, max_epoch, model_path, data, latent_dim, sample_path, log_dir, learnrate_init):

        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.saved_model_path = model_path
        self.ds_train = data
        self.latent_dim = latent_dim
        self.sample_path = sample_path
        self.log_dir = log_dir
        self.learn_rate_init = learnrate_init
        self.log_vars = []

        self.channel = 3
        self.output_size = CelebA().image_size
        self.images = tf.placeholder(tf.float32, [self.batch_size, self.output_size, self.output_size, self.channel])

        self.z_p = tf.placeholder(tf.float32, [self.batch_size , self.latent_dim])
        self.ep = tf.random_normal(shape=[self.batch_size, self.latent_dim])

    def build_model_vaegan(self):

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

        # preceptual loss(feature loss)
        self.LL_loss = tf.reduce_mean(self.NLLNormal(self.l_x_tilde, self.l_x))

        #For encode
        self.encode_loss = self.kl_loss/(128*64) - self.LL_loss

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
        new_learning_rate = tf.train.exponential_decay(self.learn_rate_init, global_step=global_step, decay_steps=20000,
                                                   decay_rate=0.98)
        #for D
        trainer_D = tf.train.RMSPropOptimizer(learning_rate=new_learning_rate)
        gradients_D = trainer_D.compute_gradients(self.D_loss, var_list=self.d_vars)
        clipped_gradients_D = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gradients_D]
        opti_D = trainer_D.apply_gradients(clipped_gradients_D)

        #for G

        trainer_G = tf.train.RMSPropOptimizer(learning_rate=new_learning_rate)
        gradients_G = trainer_G.compute_gradients(self.G_loss, var_list=self.g_vars)
        clipped_gradients_G = [(tf.clip_by_value(_[0], -1, 1.), _[1]) for _ in gradients_G]
        opti_G = trainer_G.apply_gradients(clipped_gradients_G)

        #for E
        trainer_E = tf.train.RMSPropOptimizer(learning_rate=new_learning_rate)
        gradients_E = trainer_E.compute_gradients(self.encode_loss, var_list=self.e_vars)
        clipped_gradients_E = [(tf.clip_by_value(_[0], -1, 1.), _[1]) for _ in gradients_E]
        opti_E = trainer_E.apply_gradients(clipped_gradients_E)

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            sess.run(init)
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)
            #self.saver.restore(sess, self.saved_model_path)
            batch_num = 0
            e = 0
            step = 0

            while e <= self.max_epoch:

                max_iter = len(self.ds_train)/self.batch_size - 1
                while batch_num < len(self.ds_train)/self.batch_size:

                    step = step + 1
                    train_list = CelebA.getNextBatch(self.ds_train, max_iter , batch_num, self.batch_size)
                    realbatch_array = CelebA.getShapeForData(train_list)
                    sample_z = np.random.normal(size=[self.batch_size, self.latent_dim])

                    sess.run(opti_E, feed_dict={self.images: realbatch_array})
                    #optimizaiton G
                    sess.run(opti_G, feed_dict={self.images: realbatch_array, self.z_p: sample_z})
                    # optimization D
                    sess.run(opti_D, feed_dict={self.images: realbatch_array, self.z_p: sample_z})

                    summary_str = sess.run(summary_op, feed_dict = {self.images:realbatch_array, self.z_p: sample_z})
                    summary_writer.add_summary(summary_str , step)

                    batch_num += 1

                    new_learn_rate = sess.run(new_learning_rate)
                    if new_learn_rate > 0.00005:
                        sess.run(add_global)

                    if step%20 == 0:

                        D_loss, fake_loss, encode_loss, LL_loss, kl_loss = sess.run([self.D_loss, self.G_loss, self.encode_loss, self.LL_loss, self.kl_loss/(128*64)], feed_dict={self.images: realbatch_array, self.z_p: sample_z})
                        print("EPOCH %d step %d: D: loss = %.7f G: loss=%.7f Encode: loss=%.7f LL loss=%.7f KL=%.7f" % (e, step, D_loss, fake_loss, encode_loss, LL_loss, kl_loss))

                    if np.mod(step , 200) == 1:

                        save_images(realbatch_array[0:100], [10, 10],
                                    '{}/train_{:02d}_{:04d}_r.png'.format(self.sample_path, e, step))

                    if np.mod(step , 200) == 1:

                        sample_images = sess.run(self.x_tilde, feed_dict={self.images: realbatch_array})
                        save_images(sample_images[0:100] , [10 , 10], '{}/train_{:02d}_{:04d}.png'.format(self.sample_path, e, step))
                        self.saver.save(sess , self.saved_model_path)

                e += 1
                batch_num = 0
            save_path = self.saver.save(sess , self.saved_model_path)
            print "Model saved in file: %s" % save_path

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
            conv4 = tf.reshape(conv4, [self.batch_size, -1])
            fl = lrelu(batch_normal(fully_connect(conv4, output_size=512, scope='dis_fully1'), scope='dis_bn4', reuse=reuse))
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









