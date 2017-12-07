import os
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.examples.tutorials.mnist import input_data


class Image_translation_net(object):
    def __init__(self, batch_size, z_dim):
        self.BS = batch_size
        self.z_dim = z_dim
        self.L0, self.L1, self.L2, self.L3, self.L4 = 10, 0.1, 100, 0.1, 100


    def _get_random_vector(self, mu=None, sigma=None): #mu:[z_dim], #sigma:[z_dim]
        if(mu):
            return np.random.normal(loc=mu, scale=sigma, size=[self.BS, self.z_dim]).astype(np.float32)
        else:
            return np.random.normal(size=[self.BS, self.z_dim]).astype(np.float32)


    def _get_dataset(self):
        return input_data.read_data_sets('MNIST_data', one_hot=True)


    def _encoder_pre(self, X, name, reuse=True):
        with tf.name_scope(name), tf.variable_scope(name, reuse=reuse):
            # conv1  #[BS,32,32,3]->[BS,16,16,64]
            W_conv1 = tf.get_variable('W_conv1', [5, 5, 3, 64], initializer=tf.contrib.layers.xavier_initializer_conv2d())
            b_conv1 = tf.get_variable('b_conv1', initializer=tf.constant(0.))
            z_conv1 = tf.nn.conv2d(X / 255, W_conv1, strides=[1, 2, 2, 1], padding='SAME') + b_conv1
            mean_conv1, variance_conv1 = tf.nn.moments(z_conv1, axes=[0, 1, 2])
            offset_conv1 = tf.get_variable('offset_conv1', initializer=tf.zeros([64]))
            scale_conv1 = tf.get_variable('scale_conv1', initializer=tf.ones([64]))
            bn_conv1 = tf.nn.batch_normalization(z_conv1, mean_conv1, variance_conv1, offset_conv1, scale_conv1, 1e-5)
            a_conv1 = tf.nn.leaky_relu(bn_conv1)

            self.e_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
            return a_conv1


    def _encoder_S(self, X_pred, name, reuse=True):
        with tf.name_scope(name), tf.variable_scope(name, reuse=reuse):
            # conv2  #[BS,16,16,64]->[BS,8,8,128]
            W_conv2 = tf.get_variable('W_conv2', [5, 5, 64, 128],initializer=tf.contrib.layers.xavier_initializer_conv2d())
            b_conv2 = tf.get_variable('b_conv2', initializer=tf.constant(0.))
            z_conv2 = tf.nn.conv2d(X_pred, W_conv2, strides=[1, 2, 2, 1], padding='SAME') + b_conv2
            mean_conv2, variance_conv2 = tf.nn.moments(z_conv2, axes=[0, 1, 2])
            offset_conv2 = tf.get_variable('offset_conv2', initializer=tf.zeros([128]))
            scale_conv2 = tf.get_variable('scale_conv2', initializer=tf.ones([128]))
            bn_conv2 = tf.nn.batch_normalization(z_conv2, mean_conv2, variance_conv2, offset_conv2, scale_conv2, 1e-5)
            a_conv2 = tf.nn.leaky_relu(bn_conv2)

            # conv3  #[BS,8,8,128]->[BS,8,8,256]
            W_conv3 = tf.get_variable('W_conv3', [5, 5, 128, 256],initializer=tf.contrib.layers.xavier_initializer_conv2d())
            b_conv3 = tf.get_variable('b_conv3', initializer=tf.constant(0.))
            z_conv3 = tf.nn.conv2d(a_conv2, W_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3
            mean_conv3, variance_conv3 = tf.nn.moments(z_conv3, axes=[0, 1, 2])
            offset_conv3 = tf.get_variable('offset_conv3', initializer=tf.zeros([256]))
            scale_conv3 = tf.get_variable('scale_conv3', initializer=tf.ones([256]))
            bn_conv3 = tf.nn.batch_normalization(z_conv3, mean_conv3, variance_conv3, offset_conv3, scale_conv3, 1e-5)
            a_conv3 = tf.nn.leaky_relu(bn_conv3)

            # conv4  #[BS,8,8,256]->[BS,8,8,512]
            W_conv4 = tf.get_variable('W_conv4', [5, 5, 256, 512],initializer=tf.contrib.layers.xavier_initializer_conv2d())
            b_conv4 = tf.get_variable('b_conv4', initializer=tf.constant(0.))
            z_conv4 = tf.nn.conv2d(a_conv3, W_conv4, strides=[1, 1, 1, 1], padding='SAME') + b_conv4
            mean_conv4, variance_conv4 = tf.nn.moments(z_conv4, axes=[0, 1, 2])
            offset_conv4 = tf.get_variable('offset_conv4', initializer=tf.zeros([512]))
            scale_conv4 = tf.get_variable('scale_conv4', initializer=tf.ones([512]))
            bn_conv4 = tf.nn.batch_normalization(z_conv4, mean_conv4, variance_conv4, offset_conv4, scale_conv4, 1e-5)
            a_conv4 = tf.nn.leaky_relu(bn_conv4)

            # flatten  #[BS,8,8,512]->[BS,32768]
            flatten = tf.reshape(a_conv4, [self.BS, -1])

            # fc1 #[BS,32768]->[BS,1024]
            W_fc1 = tf.get_variable('W_fc1', [flatten.shape[1].value, 1024],initializer=tf.contrib.layers.xavier_initializer())
            b_fc1 = tf.get_variable('b_fc1', [1024], initializer=tf.constant_initializer(0.))
            z_fc1 = tf.matmul(flatten, W_fc1) + b_fc1
            mean_fc1, variance_fc1 = tf.nn.moments(z_fc1, axes=[0])
            offset_fc1 = tf.get_variable('offset_fc1', initializer=tf.zeros([1024]))
            scale_fc1 = tf.get_variable('scale_fc1', initializer=tf.ones([1024]))
            bn_fc1 = tf.nn.batch_normalization(z_fc1, mean_fc1, variance_fc1, offset_fc1, scale_fc1, 0.001)
            a_fc1 = tf.nn.leaky_relu(bn_fc1)

            # fc1 #[BS,1024]->[BS,2*z_dim]
            W_fc2 = tf.get_variable('W_fc2', [1024, 2 * self.z_dim],initializer=tf.contrib.layers.xavier_initializer())
            b_fc2 = tf.get_variable('b_fc2', [2 * self.z_dim], initializer=tf.constant_initializer(0.))
            z_fc2 = tf.matmul(a_fc1, W_fc2) + b_fc2
            mean_fc2, variance_fc2 = tf.nn.moments(z_fc2, axes=[0])
            offset_fc2 = tf.get_variable('offset_fc2', initializer=tf.zeros([2 * self.z_dim]))
            scale_fc2 = tf.get_variable('scale_fc2', initializer=tf.ones([2 * self.z_dim]))
            bn_fc2 = tf.nn.batch_normalization(z_fc2, mean_fc2, variance_fc2, offset_fc2, scale_fc2, 0.001)
            a_fc2 = tf.nn.leaky_relu(bn_fc2)

            mean, stddev = tf.split(a_fc2, 2, axis=1)  # [bs, z_dim],#[bs, z_dim]
            stddev = 1e-6 + tf.nn.softplus(stddev) # [bs, z_dim]

            self.e_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
            return mean, stddev # [bs, z_dim],#[bs, z_dim]


    def _generator_S(self, z, name, reuse=True):
        with tf.name_scope(name), tf.variable_scope(name, reuse=reuse):
            # defc  # [BS,vec_size]->[BS,4*4*512]
            W_defc = tf.get_variable('W_defc', [z.shape[1].value, 4 * 4 * 1024], initializer=tf.truncated_normal_initializer(stddev=0.02))
            b_defc = tf.get_variable('b_defc', [4 * 4 * 1024], initializer=tf.constant_initializer(0.))
            z_defc1 = tf.matmul(z, W_defc) + b_defc
            # deflatten  # [BS,4*4*512]->[BS,4,4,512]
            deconv0 = tf.reshape(z_defc1, [-1, 4, 4, 1024])

            mean_conv0, variance_conv0 = tf.nn.moments(deconv0, axes=[0, 1, 2])
            offset_deconv0 = tf.get_variable('offset_deconv0', initializer=tf.zeros([1024]))
            scale_deconv0 = tf.get_variable('scale_deconv0', initializer=tf.ones([1024]))
            bn_deconv0 = tf.nn.batch_normalization(deconv0, mean_conv0, variance_conv0, offset_deconv0, scale_deconv0, 1e-5)
            a_deconv0 = tf.nn.relu(bn_deconv0)

            # deconv1  # [BS,4,4,1024]->[BS,8,8,512]
            W_deconv1 = tf.get_variable('W_deconv1', [5, 5, 512, 1024], initializer=tf.truncated_normal_initializer(stddev=0.02))
            z_deconv1 = tf.nn.conv2d_transpose(a_deconv0, W_deconv1, [self.BS, 8, 8, 512], [1, 2, 2, 1])
            mean_deconv1, variance_deconv1 = tf.nn.moments(z_deconv1, axes=[0, 1, 2])
            offset_deconv1 = tf.get_variable('offset_deconv1', initializer=tf.zeros([512]))
            scale_deconv1 = tf.get_variable('scale_deconv1', initializer=tf.ones([512]))
            bn_deconv1 = tf.nn.batch_normalization(z_deconv1, mean_deconv1, variance_deconv1, offset_deconv1, scale_deconv1, 1e-5)
            a_deconv1 = tf.nn.relu(bn_deconv1)

            # deconv2  # [BS,8,8,512]->[BS,16,16,256]
            W_deconv2 = tf.get_variable('W_deconv2', [5, 5, 256, 512], initializer=tf.truncated_normal_initializer(stddev=0.02))
            z_deconv2 = tf.nn.conv2d_transpose(a_deconv1, W_deconv2, [self.BS, 16, 16, 256], [1, 2, 2, 1])
            mean_deconv2, variance_deconv2 = tf.nn.moments(z_deconv2, axes=[0, 1, 2])
            offset_deconv2 = tf.get_variable('offset_deconv2', initializer=tf.zeros([256]))
            scale_deconv2 = tf.get_variable('scale_deconv2', initializer=tf.ones([256]))
            bn_deconv2 = tf.nn.batch_normalization(z_deconv2, mean_deconv2, variance_deconv2, offset_deconv2, scale_deconv2, 1e-5)
            a_deconv2 = tf.nn.relu(bn_deconv2)

            # deconv3  # [BS,16,16,256]->[BS,32,32,128]
            W_deconv3 = tf.get_variable('W_deconv3', [5, 5, 128, 256], initializer=tf.truncated_normal_initializer(stddev=0.02))
            z_deconv3 = tf.nn.conv2d_transpose(a_deconv2, W_deconv3, [self.BS, 32, 32, 128], [1, 2, 2, 1])
            mean_deconv3, variance_deconv3 = tf.nn.moments(z_deconv3, axes=[0, 1, 2])
            offset_deconv3 = tf.get_variable('offset_deconv3', initializer=tf.zeros([128]))
            scale_deconv3 = tf.get_variable('scale_deconv3', initializer=tf.ones([128]))
            bn_deconv3 = tf.nn.batch_normalization(z_deconv3, mean_deconv3, variance_deconv3, offset_deconv3, scale_deconv3, 1e-5)
            a_deconv3 = tf.nn.relu(bn_deconv3)

        self.g_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        return a_deconv3


    def _generator_end(self, recon_X_shared, name, reuse=True):
        with tf.name_scope(name), tf.variable_scope(name, reuse=reuse):
            # deconv4  # [BS,32,32,128]->[BS,64,64,64]
            W_deconv4 = tf.get_variable('W_deconv4', [5, 5, 64, 128], initializer=tf.truncated_normal_initializer(stddev=0.02))
            z_deconv4 = tf.nn.conv2d_transpose(recon_X_shared, W_deconv4, [self.BS, 64, 64, 64], [1, 2, 2, 1])
            mean_deconv4, variance_deconv4 = tf.nn.moments(z_deconv4, axes=[0, 1, 2])
            offset_deconv4 = tf.get_variable('offset_deconv4', initializer=tf.zeros([64]))
            scale_deconv4 = tf.get_variable('scale_deconv4', initializer=tf.ones([64]))
            bn_deconv4 = tf.nn.batch_normalization(z_deconv4, mean_deconv4, variance_deconv4, offset_deconv4, scale_deconv4,1e-5)
            a_deconv4 = tf.nn.relu(bn_deconv4)

            # deconv5  # [BS,64,64,64]->[BS,128,128,3]
            W_deconv5 = tf.get_variable('W_deconv5', [5, 5, 3, 64], initializer=tf.truncated_normal_initializer(stddev=0.02))
            z_deconv5 = tf.nn.conv2d_transpose(a_deconv4, W_deconv5, [self.BS, 128, 128, 3], [1, 2, 2, 1])

            self.g_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
            return z_deconv5


    def _discriminator_pre(self, recon_X, name, reuse=True):
        with tf.name_scope(name), tf.variable_scope(name, reuse=reuse):
            # conv1  #[BS,128,128,3]->[BS,64,64,64]
            W_conv1 = tf.get_variable('W_conv1', [5, 5, 3, 64], initializer=tf.truncated_normal_initializer(stddev=0.02))
            b_conv1 = tf.get_variable('b_conv1', initializer=tf.constant(0.))
            z_conv1 = tf.nn.conv2d(recon_X / 255, W_conv1, strides=[1, 2, 2, 1], padding='SAME') + b_conv1
            mean_conv1, variance_conv1 = tf.nn.moments(z_conv1, axes=[0, 1, 2])
            offset_conv1 = tf.get_variable('offset_conv1', initializer=tf.zeros([64]))
            scale_conv1 = tf.get_variable('scale_conv1', initializer=tf.ones([64]))
            bn_conv1 = tf.nn.batch_normalization(z_conv1, mean_conv1, variance_conv1, offset_conv1, scale_conv1, 1e-5)
            a_conv1 = tf.nn.leaky_relu(bn_conv1)
            self.d_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
            return a_conv1


    def _discriminator_S(self, im_pred, name, reuse=True):
        with tf.name_scope(name), tf.variable_scope(name, reuse=reuse):
            # conv2  #[BS,64,64,64]->[BS,32,32,128]
            W_conv2 = tf.get_variable('W_conv2', [5, 5, 64, 128], initializer=tf.truncated_normal_initializer(stddev=0.02))
            b_conv2 = tf.get_variable('b_conv2', initializer=tf.constant(0.))
            z_conv2 = tf.nn.conv2d(im_pred, W_conv2, strides=[1, 2, 2, 1], padding='SAME') + b_conv2
            mean_conv2, variance_conv2 = tf.nn.moments(z_conv2, axes=[0, 1, 2])
            offset_conv2 = tf.get_variable('offset_conv2', initializer=tf.zeros([128]))
            scale_conv2 = tf.get_variable('scale_conv2', initializer=tf.ones([128]))
            bn_conv2 = tf.nn.batch_normalization(z_conv2, mean_conv2, variance_conv2, offset_conv2, scale_conv2, 1e-5)
            a_conv2 = tf.nn.leaky_relu(bn_conv2)

            # conv3  #[BS,32,32,128]->[BS,16,16,256]
            W_conv3 = tf.get_variable('W_conv3', [5, 5, 128, 256], initializer=tf.truncated_normal_initializer(stddev=0.02))
            b_conv3 = tf.get_variable('b_conv3', initializer=tf.constant(0.))
            z_conv3 = tf.nn.conv2d(a_conv2, W_conv3, strides=[1, 2, 2, 1], padding='SAME') + b_conv3
            mean_conv3, variance_conv3 = tf.nn.moments(z_conv3, axes=[0, 1, 2])
            offset_conv3 = tf.get_variable('offset_conv3', initializer=tf.zeros([256]))
            scale_conv3 = tf.get_variable('scale_conv3', initializer=tf.ones([256]))
            bn_conv3 = tf.nn.batch_normalization(z_conv3, mean_conv3, variance_conv3, offset_conv3, scale_conv3, 1e-5)
            a_conv3 = tf.nn.leaky_relu(bn_conv3)

            # conv4  #[BS,16,16,256]->[BS,8,8,512]
            W_conv4 = tf.get_variable('W_conv4', [5, 5, 256, 512], initializer=tf.truncated_normal_initializer(stddev=0.02))
            b_conv4 = tf.get_variable('b_conv4', initializer=tf.constant(0.))
            z_conv4 = tf.nn.conv2d(a_conv3, W_conv4, strides=[1, 2, 2, 1], padding='SAME') + b_conv4
            mean_conv4, variance_conv4 = tf.nn.moments(z_conv4, axes=[0, 1, 2])
            offset_conv4 = tf.get_variable('offset_conv4', initializer=tf.zeros([512]))
            scale_conv4 = tf.get_variable('scale_conv4', initializer=tf.ones([512]))
            bn_conv4 = tf.nn.batch_normalization(z_conv4, mean_conv4, variance_conv4, offset_conv4, scale_conv4, 1e-5)
            a_conv4 = tf.nn.leaky_relu(bn_conv4)

            # flatten  #[BS,8,8,512]->[BS,32768]
            flatten = tf.reshape(a_conv4, [self.BS, 32768])

            # fc1 # classify  #[BS,32768]->[BS,1]
            W_fc1 = tf.get_variable('W_fc1', [flatten.shape[1].value, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
            b_fc1 = tf.get_variable('b_fc1', [1], initializer=tf.constant_initializer(0.))
            logits = tf.matmul(flatten, W_fc1) + b_fc1

            self.d_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
            return logits


    def build_graph(self):
        # placeholder
        self.XA = tf.placeholder(tf.float32, [self.BS, 28, 28 ,3]) #[BS,W,H,C]
        self.XB = tf.placeholder(tf.float32, [self.BS, 28, 28 ,3]) #[BS,W,H,C]
        # net
        pre_msA = self._encoder_pre(self.XA, 'encoder_A')
        pre_msB = self._encoder_pre(self.XB, 'encoder_B')

        self.muA, self.sigmaA = self._encoder_S(pre_msA, 'encoder_S') # [BS,z_dim],#[BS,z_dim]
        self.muB, self.sigmaB = self._encoder_S(pre_msB, 'encoder_S') # [BS,z_dim],#[BS,z_dim]

        self.zA = self.muA + self.sigmaA * tf.random_normal(tf.shape(self.muA), 0, 1, dtype=tf.float32)  # [BS,z_dim]
        self.zB = self.muB + self.sigmaB * tf.random_normal(tf.shape(self.muB), 0, 1, dtype=tf.float32)  # [BS,z_dim]

        sR_A = self._generator_S(self.zA, 'generator_S')
        sR_B = self._generator_S(self.zB, 'generator_S')

        self.RA_A = self._generator_end(sR_A, 'generator_A') #[BS,W,H,C]
        self.RA_B = self._generator_end(sR_B, 'generator_A') #[BS,W,H,C]

        self.RB_B = self._generator_end(sR_B, 'generator_B') #[BS,W,H,C]
        self.RB_A = self._generator_end(sR_A, 'generator_B') #[BS,W,H,C]

        pre_yA_A = self._discriminator_pre(self.RA_A, 'discriminator_A')
        pre_yA_B = self._discriminator_pre(self.RA_B, 'discriminator_A')
        yA_A = self._discriminator_S(pre_yA_A, 'discriminator_S') #[BS,1]
        yA_B = self._discriminator_S(pre_yA_B, 'discriminator_S') #[BS,1]

        pre_yB_B = self._discriminator_pre(self.RB_B, 'discriminator_B')
        pre_yB_A = self._discriminator_pre(self.RB_A, 'discriminator_B')
        yB_B = self._discriminator_S(pre_yB_B, 'discriminator_S') #[BS,1]
        yB_A = self._discriminator_S(pre_yB_A, 'discriminator_S') #[BS,1]

        #cycle_net
        XC = self.RB_A
        XD = self.RA_B

        pre_msC= self._encoder_pre(XC, 'encoder_B')
        pre_msD = self._encoder_pre(XD, 'encoder_A')

        self.muC, self.sigmaC = self._encoder_S(pre_msC, 'encoder_S') # [BS,z_dim],#[BS,z_dim]
        self.muD, self.sigmaD = self._encoder_S(pre_msD, 'encoder_S') # [BS,z_dim],#[BS,z_dim]

        self.zC = self.muC + self.sigmaC * tf.random_normal(tf.shape(self.muC), 0, 1, dtype=tf.float32)  # [BS,z_dim]
        self.zD = self.muD + self.sigmaD * tf.random_normal(tf.shape(self.muD), 0, 1, dtype=tf.float32)  # [BS,z_dim]

        sRC = self._generator_S(self.zC, 'generator_S')
        sRD = self._generator_S(self.zD, 'generator_S')

        self.RC = self._generator_end(sRC, 'generator_A') #RA_(RB_A)
        self.RD = self._generator_end(sRD, 'generator_B') #RB_(RA_B)

        #VAE_loss
        KL_lossA = 0.5 * tf.reduce_sum(tf.square(self.muA) + tf.square(self.sigmaA) - tf.log(1e-8 + tf.square(self.sigmaA)) - 1, [1])# [BS,z_dim]->[BS,1]
        IO_lossA = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.RA_A, labels=self.XA), [1, 2, 3])# [BS,w,h,c]->[BS,1]
        VAE_lossA = tf.reduce_mean(self.L1*KL_lossA + self.L2*IO_lossA) # [1] #Optimize(EA,GA)

        KL_lossB = 0.5 * tf.reduce_sum(tf.square(self.muB) + tf.square(self.sigmaB) - tf.log(1e-8 + tf.square(self.sigmaB)) - 1, [1])# [BS,z_dim]->[BS,1]
        IO_lossB = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.RB_B, labels=self.XB), [1, 2, 3])# [BS,w,h,c]->[BS,1]
        VAE_lossB = tf.reduce_mean(self.L1*KL_lossB + self.L2*IO_lossB) # [1] #Optimize(EB,GB)

        self.VAE_loss = VAE_lossA + VAE_lossB # [1]

        #GAN_loss
        GAN_lossA_A = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=yA_A, labels=tf.ones_like(yA_A))) #real
        GAN_lossA_B = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=yA_B, labels=tf.zeros_like(yA_B))) #fake
        GAN_lossA = GAN_lossA_A + GAN_lossA_B  # [1] #Optimize(EA,GA,DA)

        GAN_lossB_B = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=yB_B, labels=tf.ones_like(yB_B)))  #real
        GAN_lossB_A = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=yB_A, labels=tf.zeros_like(yB_A)))  #fake
        GAN_lossB = GAN_lossB_B + GAN_lossB_A  # [1] #Optimize(EB,GB,DB)

        self.GAN_loss = self.L0*(GAN_lossA + GAN_lossB) # [1]

        # Cycle_loss #Optimize(EA,GA,EB,GB)
        KL_lossC = 0.5 * tf.reduce_sum(tf.square(self.muC) + tf.square(self.sigmaC) - tf.log(1e-8 + tf.square(self.sigmaC)) - 1, [1])  # [BS,z_dim]->[BS,1]
        CIO_lossA = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.RC, labels=self.XA), [1, 2, 3])# [BS,w,h,c]->[BS,1]
        Cycle_lossA = tf.reduce_mean(self.L3*KL_lossA + self.L3*KL_lossC + self.L4*CIO_lossA) #[BS,1]->[1]

        KL_lossD = 0.5 * tf.reduce_sum(tf.square(self.muD) + tf.square(self.sigmaD) - tf.log(1e-8 + tf.square(self.sigmaD)) - 1, [1])  # [BS,z_dim]->[BS,1]
        CIO_lossB = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.RD, labels=self.XB), [1, 2, 3])# [BS,w,h,c]->[BS,1]
        Cycle_lossB = tf.reduce_mean(self.L3*KL_lossB + self.L3*KL_lossD + self.L4*CIO_lossB) #[BS,1]->[1]

        self.Cycle_loss = Cycle_lossA + Cycle_lossB

        #loss
        self.loss = self.VAE_loss + self.GAN_loss + self.Cycle_loss # [1]

        # optimizers
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(self.loss)

        #tensorboard
        self.sum_VAE_loss = tf.summary.scalar("VAE_loss", self.VAE_loss)
        self.sum_GAN_loss = tf.summary.scalar("GAN_loss", self.GAN_loss)
        self.sum_Cycle_loss = tf.summary.scalar("Cycle_loss", self.Cycle_loss)
        self.sum_loss = tf.summary.scalar("loss", self.loss)
        self.sum_merge = tf.summary.merge_all()


    def train(self):
        tf_sum_writer = tf.summary.FileWriter('logs')

        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir='tfModel/')

        mnist = self._get_dataset()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            if ckpt and ckpt.model_checkpoint_path:
                print('loading_model')
                saver.restore(sess, ckpt.model_checkpoint_path)
                pre_model_epoch = int(ckpt.model_checkpoint_path[13:])
                print('pre_model_epoch:', pre_model_epoch)
            else:
                pre_model_epoch = 0
                print('no_pre_model')

            tf_sum_writer.add_graph(sess.graph)
            global_step = 0
            for epoch in range(pre_model_epoch + 1, pre_model_epoch + 500):
                for epoch_step in range(150):
                    XA, XB = self._get_dataset()
                    #train
                    _, sum_merge, loss, VAE_loss, GAN_loss, Cycle_loss = sess.run(
                        [self.optimizer, self.sum_merge, self.loss, self.VAE_loss, self.GAN_loss, self.Cycle_loss],
                        feed_dict={self.XA: XA, self.XB: XB})

                    if epoch_step % 10 == 0: # tensorboard
                        tf_sum_writer.add_summary(sum_merge, global_step=global_step)

                    print('epoch:', epoch, 'epoch_step:', epoch_step, 'global_step:', global_step)
                    global_step = global_step + 1

                if epoch % 50 == 0: # save model
                    print('---------------------')
                    if not os.path.exists('./tfModel/'):
                        os.makedirs('./tfModel/')
                    saver.save(sess, './tfModel/epoch' + str(epoch))

                if epoch % 10 == 0: # save images
                    A2B_path = './generated_image/epoch' + str(epoch) + '/A2B'
                    B2A_path = './generated_image/epoch' + str(epoch) + '/A2B'
                    if not os.path.exists(A2B_path):
                        os.makedirs(A2B_path)
                    if not os.path.exists(B2A_path):
                        os.makedirs(B2A_path)
                    XA, XB = self._get_dataset()
                    RA_B, RB_A = sess.run([self.RA_B, self.RB_A],feed_dict={self.XA: XA, self.XB: XB})
                    RA_B, RB_A = RA_B * 255.0, RB_A*255.0
                    RA_B.astype(np.uint8)
                    RB_A.astype(np.uint8)
                    for i in range(self.BS):
                        cv2.imwrite(A2B_path + '/A' + str(i) + '.jpg', XA[i])
                        cv2.imwrite(A2B_path + '/RB_A' + str(i) + '.jpg', RB_A[i])
                        cv2.imwrite(B2A_path + '/B' + str(i) + '.jpg', XB[i])
                        cv2.imwrite(B2A_path + '/RA_B' + str(i) + '.jpg', RA_B[i])


if __name__ == "__main__":
    vae = Image_translation_net(32, 100)
    vae.build_graph()
    vae.train()