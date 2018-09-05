
import tensorflow as tf
from tensorflow.contrib import slim


class Lenet(object):
    def __init__(self, inputs, scope='lenet', training_flag=True, reuse=False):
        self.scope=scope
        self.inputs=inputs
        if inputs.get_shape()[3] == 3:
            self.inputs = tf.image.rgb_to_grayscale(self.inputs)
        self.training_flag=training_flag
        self.is_training=True
        self.reuse=reuse
        self.create()


    def create(self,is_training=False):

        with tf.variable_scope(self.scope, reuse=self.reuse):
            with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
                with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, padding='VALID'):
                    net=self.inputs
                    net = slim.conv2d(net, 64, 5, scope='conv1')
                    self.conv1=net
                    net = slim.max_pool2d(net, 2, stride=2, scope='pool1')
                    self.pool1 = net
                    net = slim.conv2d(net,128, 5, scope='conv2')
                    self.conv2= net
                    net = slim.max_pool2d(net, 2, stride=2, scope='pool2')
                    self.pool2= net
                    net = tf.contrib.layers.flatten(net)
                    net = slim.fully_connected(net, 1024, activation_fn=tf.nn.relu, scope='fc3')
                    self.fc3= net
                    net = slim.dropout(net,0.5, is_training=self.training_flag)
                    net = slim.fully_connected(net,64, activation_fn=tf.nn.relu ,scope='fc4')
                    self.fc4 = net
                    net = slim.fully_connected(net,10, activation_fn=None, scope='fc5')
                    self.fc5 = net
                    self.softmax_output=slim.softmax(net,scope='prediction')

