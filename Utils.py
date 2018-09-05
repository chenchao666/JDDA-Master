import random
import tensorflow as tf
from functools import partial
from numpy import *




def shuffle0(Data, Label):
    ind = range(Data.shape[0])
    random.shuffle(ind)
    Data = Data[ind, :, :, :]
    Label = Label[ind, :]
    return Data, Label



def shuffle(Data, Label, Weights):
    ind = range(Data.shape[0])
    random.shuffle(ind)
    Data = Data[ind, :, :, :]
    Label = Label[ind, :]
    Weights=Weights[ind,:]
    Weights=Weights[:,ind]
    return Data, Label, Weights

def compute_pairwise_distances(x, y):
    if not len(x.get_shape()) == len(y.get_shape()) == 2:
        raise ValueError('Both inputs should be matrices.')
    if x.get_shape().as_list()[1] != y.get_shape().as_list()[1]:
        raise ValueError('The number of features should be the same.')

    norm = lambda x: tf.reduce_sum(tf.square(x), 1)
    return tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))


def gaussian_kernel_matrix(x, y, sigmas):
    beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))
    dist = compute_pairwise_distances(x, y)
    s = tf.matmul(beta, tf.reshape(dist, (1, -1)))
    return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))


def maximum_mean_discrepancy(x, y, kernel=gaussian_kernel_matrix):
    cost = tf.reduce_mean(kernel(x, x))
    cost += tf.reduce_mean(kernel(y, y))
    cost -= 2 * tf.reduce_mean(kernel(x, y))
    # We do not allow the loss to become negative.
    cost = tf.where(cost > 0, cost, 0, name='value')
    return cost


def KMMD(Xs,Xt):
    # sigmas=[1e-2,0.1,1,5,10,20,25,30,35,100]
    # guassian_kernel=partial(kernel,sigmas=tf.constant(sigmas))
    # cost = tf.reduce_mean(guassian_kernel(Xs, Xs))
    # cost += tf.reduce_mean(guassian_kernel(Xt, Xt))
    # cost -= 2 * tf.reduce_mean(guassian_kernel(Xs, Xt))
    # cost = tf.where(cost > 0, cost, 0)

    sigmas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6]
    gaussian_kernel = partial(gaussian_kernel_matrix, sigmas=tf.constant(sigmas))
    cost= maximum_mean_discrepancy(Xs, Xt, kernel=gaussian_kernel)


    return cost

def kernel(X, Y, sigmas):
    beta = 1.0/(2.0 * (tf.expand_dims(sigmas,1)))
    dist = Cal_pairwise_dist(X,Y)
    s = tf.matmul(beta, tf.reshape(dist,(1,-1)))
    return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))



def Cal_pairwise_dist(X,Y):
    norm = lambda x: tf.reduce_sum(tf.square(x), 1)
    dist= tf.transpose(norm(tf.expand_dims(X, 2)-tf.transpose(Y)))
    return dist


def mmatch(x1,x2,n_moments):
    mx1=tf.reduce_mean(x1,axis=0)
    mx2=tf.reduce_mean(x2,axis=0)
    sx1=x1-mx1
    sx2=x2-mx2
    dm=matchnorm(mx1,mx2)
    scms=dm
    for i in range(n_moments-1):
        scms+=scm(sx1,sx2,i+2)
    return scms

def matchnorm(x1,x2):
    return tf.sqrt(tf.reduce_sum((x1-x2)**2))

def scm(sx1,sx2,k):
    ss1=tf.reduce_mean(sx1**k,axis=0)
    ss2=tf.reduce_mean(sx2**k,axis=0)
    return matchnorm(ss1,ss2)


def Label2EdgeWeights(Label):
    Label=Label+1
    Label.astype("float32")
    n=Label.shape
    n=n[0]
    EdgeWeights=zeros((n,n))
    Label=expand_dims(Label,axis=0)
    EdgeWeights=transpose(1.0/Label)*Label
    indx,indy=where(EdgeWeights!=1.0)
    EdgeWeights[indx,indy]=0.
    return EdgeWeights



def symmetric_matrix_square_root(mat,eps=1e-8):
    s,u,v=tf.svd(mat)
    si=tf.where(tf.less(s,eps),s,tf.sqrt(s))
    return tf.matmul(tf.matmul(u,tf.diag(si)),v,transpose_b=True)




























