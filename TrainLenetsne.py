import os
import pickle
from Lenet import *
from Utils import *
import scipy.io
import numpy as np
from tensorflow.contrib import slim
from center_loss import *
#
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.manifold import TSNE
os.environ['CUDA_VISIBLE_DEVICES']='1'



class Train():
    def __init__(self,class_num,batch_size,iters,learning_rate,keep_prob,param):
        self.ClassNum=class_num
        self.BatchSize=batch_size
        self.Iters=iters
        self.LearningRate=learning_rate
        self.KeepProb=keep_prob
        self.discriminative_loss_param=param[0]
        self.domain_loss_param=param[1]
        self.adver_loss_param=param[2]

        self.SourceData,self.SourceLabel=load_svhn('svhn')
        self.TargetData, self.TargetLabel=load_mnist('mnist')
        self.TestData, self.TestLabel = load_mnist('mnist',split='test')
        # self.EdgeWeights=Label2EdgeWeights(self.SourceLabel)
        self.EdgeWeights=zeros((self.SourceLabel.shape[0],self.SourceLabel.shape[0]))


        #######################################################################################
        self.source_image = tf.placeholder(tf.float32, shape=[None, 32,32,3],name="source_image")
        self.source_label = tf.placeholder(tf.float32, shape=[None, self.ClassNum],name="source_label")
        self.target_image = tf.placeholder(tf.float32, shape=[None, 32, 32,1],name="target_image")
        self.Training_flag = tf.placeholder(tf.bool, shape=None,name="Training_flag")
        self.W = tf.placeholder(tf.float32, shape=[self.BatchSize, self.BatchSize])


    def TrainNet(self):
        self.source_model=Lenet(inputs=self.source_image, training_flag=self.Training_flag, reuse=False)
        self.target_model=Lenet(inputs=self.target_image, training_flag=self.Training_flag, reuse=True)
        self.CalLoss()
        varall=tf.trainable_variables()
        with tf.control_dependencies([self.centers_update_op]):
            self.solver = tf.train.AdamOptimizer(learning_rate=self.LearningRate).minimize(self.loss)
        self.source_prediction = tf.argmax(self.source_model.softmax_output, 1)
        self.target_prediction = tf.argmax(self.target_model.softmax_output, 1)
        with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            self.SourceLabel=sess.run(tf.one_hot(self.SourceLabel,10))
            self.TestLabel=sess.run(tf.one_hot(self.TestLabel,10))
            true_num = 0.0
            lossData=[]
            for step in range(self.Iters):
                i= step % int(self.SourceData.shape[0]/self.BatchSize)
                j= step % int(self.TargetData.shape[0]/self.BatchSize)
                source_batch_x = self.SourceData[i * self.BatchSize: (i + 1) * self.BatchSize]
                source_batch_y = self.SourceLabel[i * self.BatchSize: (i + 1) * self.BatchSize]
                target_batch_x = self.TargetData[j * self.BatchSize: (j + 1) * self.BatchSize]
                W = self.EdgeWeights[i * self.BatchSize: (i + 1) * self.BatchSize,i * self.BatchSize: (i + 1) * self.BatchSize]
                total_loss, source_loss,domain_loss,intra_loss,inter_loss, source_prediction,_= sess.run(
                    fetches=[self.loss, self.source_loss, self.domain_loss, self.intra_loss, self.inter_loss, self.source_prediction, self.solver],
                    feed_dict={self.source_image: source_batch_x, self.source_label: source_batch_y,self.target_image: target_batch_x, self.Training_flag: True, self.W: W})

                true_label = argmax(source_batch_y, 1)
                true_num = true_num + sum(true_label == source_prediction)


                if step % 200 ==0:
                    print "Iters-{} ### TotalLoss={} ### SourceLoss={} ###DomainLoss={} ## IntraLoss={} ### InterLoss={} ".format(step, total_loss, source_loss, domain_loss, intra_loss, inter_loss)
                    train_accuracy = true_num / (200*self.BatchSize)
                    true_num = 0.0
                    print " ########## train_accuracy={} ###########".format(train_accuracy)
                    self.Test(sess,lossData)
            # savedata=np.array(lossData)
            # np.save("MNISTtoMNISTSOU.npy",savedata)
                if step % 2000 == 0:
                    pass
                    # savedata = np.array(lossData)
                    # np.save("SVHNtoMNIST.npy", savedata)

                    # self.conputeTSNE(step, self.SourceData, self.TargetData, self.SourceLabel, self.TargetLabel,
                    #                  sess)
                    # self.SourceData,self.SourceLabel,self.EdgeWeights=shuffle(self.SourceData,self.SourceLabel,self.EdgeWeights)
                    # print("success")


                # if step!=0 and step % 1000 == 0:
                #     self.SourceData, self.SourceLabel= shuffle0(self.SourceData, self.SourceLabel)

                # if step!=0 and step % 5000 == 0:
                #    self.SourceData, self.SourceLabel, self.EdgeWeights= shuffle(self.SourceData, self.SourceLabel, self.EdgeWeights)




    def CalLoss(self):
        self.source_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.source_label, logits=self.source_model.fc5)
        self.source_loss = tf.reduce_mean(self.source_cross_entropy)
        self.CalDiscriminativeLoss(method="CenterBased")
        self.CalDomainLoss(method="CORAL")
        # self.CalAdver()
        self.loss=self.source_loss+self.domain_loss_param*self.domain_loss+self.discriminative_loss_param*self.discriminative_loss



    def CalDomainLoss(self,method):
        if method=="MMD":
            Xs=self.source_model.fc4
            Xt=self.target_model.fc4
            diff=tf.reduce_mean(Xs, 0, keep_dims=False) - tf.reduce_mean(Xt, 0, keep_dims=False)
            self.domain_loss=tf.reduce_sum(tf.multiply(diff,diff))


        elif method=="KMMD":
            Xs=self.source_model.fc4
            Xt=self.target_model.fc4
            self.domain_loss=tf.maximum(0.0001,KMMD(Xs,Xt))


        elif method=="CORAL":
            Xs = self.source_model.fc4
            Xt = self.target_model.fc4
            self.domain_loss=self.coral_loss(Xs,Xt)


        elif method =='LCORAL':
            Xs = self.source_model.fc4
            Xt = self.target_model.fc4
            self.domain_loss=self.log_coral_loss(Xs,Xt)

        elif method == 'SteinCORAL':
            Xs = self.source_model.fc4
            Xt = self.target_model.fc4
            self.domain_loss=self.SteinCoral_loss(Xs, Xt)
            # self.domain_loss = self.AIRMCoral_loss(Xs, Xt)


        elif method == 'JeffCORAL':
            Xs = self.source_model.fc4
            Xt = self.target_model.fc4
            self.domain_loss = self.JeffCoral_loss(Xs, Xt)



        elif method =="mmatch":
            Xs = self.source_model.fc4
            Xt = self.target_model.fc4
            self.domain_loss=mmatch(Xs,Xt,5)



    def CalTargetLoss(self,method):
        if method=="Entropy":
            trg_softmax=self.target_model.softmax_output
            self.target_loss=-tf.reduce_mean(tf.reduce_sum(trg_softmax * tf.log(trg_softmax), axis=1))



    def CalDiscriminativeLoss(self,method):
        if method=="InstanceBased":
            Xs = self.source_model.fc4
            norm = lambda x: tf.reduce_sum(tf.square(x), 1)
            self.F0 = tf.transpose(norm(tf.expand_dims(Xs, 2) - tf.transpose(Xs)))
            margin0 = 0
            margin1 = 100
            F0=tf.pow(tf.maximum(0.0, self.F0-margin0),2)
            F1=tf.pow(tf.maximum(0.0, margin1-self.F0),2)
            self.intra_loss=tf.reduce_mean(tf.multiply(F0, self.W))
            self.inter_loss=tf.reduce_mean(tf.multiply(F1, 1.0-self.W))
            self.discriminative_loss = (self.intra_loss+self.inter_loss) / (self.BatchSize * self.BatchSize)



        elif method=="CenterBased":
            Xs=self.source_model.fc4
            labels=tf.argmax(self.source_label,1)
            self.inter_loss, self.intra_loss, self.centers_update_op = get_center_loss(Xs, labels, 0.5, 10)
            self.discriminative_loss = self.intra_loss+ self.inter_loss
            self.discriminative_loss=self.discriminative_loss/(self.ClassNum*self.BatchSize+self.ClassNum*self.ClassNum)




    def coral_loss(self, h_src, h_trg, gamma=1e-3):
        # regularized covariances (D-Coral is not regularized actually..)
        # First: subtract the mean from the data matrix
        batch_size = self.BatchSize
        h_src = h_src - tf.reduce_mean(h_src, axis=0)
        h_trg = h_trg - tf.reduce_mean(h_trg, axis=0)
        cov_source = (1. / (batch_size - 1)) * tf.matmul(h_src, h_src,
                                                         transpose_a=True)  # + gamma * tf.eye(self.hidden_repr_size)
        cov_target = (1. / (batch_size - 1)) * tf.matmul(h_trg, h_trg,
                                                         transpose_a=True)  # + gamma * tf.eye(self.hidden_repr_size)
        # cov_source=tf.linalg.cholesky(cov_source)
        # cov_target=tf.linalg.cholesky(cov_target)
        return tf.reduce_mean(tf.square(tf.subtract(cov_source, cov_target)))





    def log_coral_loss(self, h_src, h_trg, gamma=1e-3):
        # regularized covariances result in inf or nan
        # First: subtract the mean from the data matrix
        batch_size = float(self.BatchSize)
        h_src = h_src - tf.reduce_mean(h_src, axis=0)
        h_trg = h_trg - tf.reduce_mean(h_trg, axis=0)
        cov_source = (1. / (batch_size - 1)) * tf.matmul(h_src, h_src,transpose_a=True)  # + gamma * tf.eye(self.hidden_repr_size)
        cov_target = (1. / (batch_size - 1)) * tf.matmul(h_trg, h_trg,transpose_a=True)  # + gamma * tf.eye(self.hidden_repr_size)
        # eigen decomposition
        eig_source = tf.self_adjoint_eig(cov_source)
        eig_target = tf.self_adjoint_eig(cov_target)
        log_cov_source = tf.matmul(eig_source[1],tf.matmul(tf.diag(tf.log(eig_source[0])), eig_source[1], transpose_b=True))
        log_cov_target = tf.matmul(eig_target[1],tf.matmul(tf.diag(tf.log(eig_target[0])), eig_target[1], transpose_b=True))
        return tf.reduce_mean(tf.square(tf.subtract(log_cov_source, log_cov_target)))



    def SteinCoral_loss(self,h_src, h_trg):
        batch_size = self.BatchSize
        h_src = h_src - tf.reduce_mean(h_src, axis=0)
        h_trg = h_trg - tf.reduce_mean(h_trg, axis=0)
        cov_source = (1. / (batch_size - 1)) * tf.matmul(h_src, h_src, transpose_a=True)+ 5.0 * tf.eye(int(h_src.shape[1]))
        cov_target = (1. / (batch_size - 1)) * tf.matmul(h_trg, h_trg, transpose_a=True)+ 5.0 * tf.eye(int(h_trg.shape[1]))
        loss=tf.linalg.logdet(0.5*(cov_source + cov_target))-0.5*tf.linalg.logdet(tf.matmul(cov_source,cov_target))
        return loss



    def JeffCoral_loss(self, h_src, h_trg):
        batch_size = self.BatchSize
        h_src = h_src - tf.reduce_mean(h_src, axis=0)
        h_trg = h_trg - tf.reduce_mean(h_trg, axis=0)
        cov_source = (1. / (batch_size - 1)) * tf.matmul(h_src, h_src, transpose_a=True) + 5.0 * tf.eye(int(h_src.shape[1]))
        cov_target = (1. / (batch_size - 1)) * tf.matmul(h_trg, h_trg, transpose_a=True) + 5.0 * tf.eye(int(h_trg.shape[1]))
        loss=0.5*tf.trace(tf.matmul(cov_source,tf.linalg.inv(cov_target))+tf.matmul(cov_target,tf.linalg.inv(cov_source)))-tf.cast(h_src.shape[1],tf.float32)
        return tf.abs(loss)



    def AIRMCoral_loss(self, h_src, h_trg):
        batch_size = self.BatchSize
        h_src = h_src - tf.reduce_mean(h_src, axis=0)
        h_trg = h_trg - tf.reduce_mean(h_trg, axis=0)
        cov_source = (1. / (batch_size - 1)) * tf.matmul(h_src, h_src, transpose_a=True) + 0.0 * tf.eye(int(h_src.shape[1]))
        cov_target = (1. / (batch_size - 1)) * tf.matmul(h_trg, h_trg, transpose_a=True) + 1.0 * tf.eye(int(h_trg.shape[1]))
        mat0=tf.matmul(cov_source,tf.linalg.inv(cov_target))
        eig_mat=tf.self_adjoint_eig(mat0)
        log_mat=tf.matmul(eig_mat[1],tf.matmul(tf.diag(tf.log(eig_mat[0])), eig_mat[1], transpose_b=True))
        return tf.reduce_mean(tf.square(log_mat))





    def Test(self,sess,lossData):
        true_num=0.0
        num = int(self.TestData.shape[0] / self.BatchSize)
        total_num=num*self.BatchSize
        for i in range (num):
            k = i % int(self.TestData.shape[0] / self.BatchSize)
            target_batch_x = self.TestData[k * self.BatchSize: (k + 1) * self.BatchSize]
            target_batch_y= self.TestLabel[k * self.BatchSize: (k + 1) * self.BatchSize]
            prediction=sess.run(fetches=self.target_prediction, feed_dict={self.target_image:target_batch_x, self.Training_flag: False})
            true_label = argmax(target_batch_y, 1)
            true_num+=sum(true_label==prediction)
        accuracy=true_num / total_num
        lossData.append(accuracy)
        print "###########  Test Accuracy={} ##########".format(accuracy)

    def conputeTSNE(self,step,source_images, target_images,source_labels,target_labels,sess):

        target_images = target_images[:2000]
        target_labels = target_labels[:2000]
        source_images = source_images[:2000]
        source_labels = source_labels[:2000]

        target_labels = one_hot(target_labels.astype(int), 10)
        print(source_labels.shape)

        assert len(target_labels) == len(source_labels)



        n_slices = int(2000 / 128)

        fx_src = np.empty((0, 64))
        fx_trg = np.empty((0, 64))

        for src_im, trg_im in zip(np.array_split(source_images, n_slices),
                                  np.array_split(target_images, n_slices),
                                  ):
            feed_dict = {self.source_image: src_im, self.target_image: trg_im,self.Training_flag:False}

            fx_src_, fx_trg_ = sess.run([self.source_model.fc4, self.target_model.fc4], feed_dict)

            fx_src = np.vstack((fx_src, np.squeeze(fx_src_)))
            fx_trg = np.vstack((fx_trg, np.squeeze(fx_trg_)))

        src_labels = np.argmax(source_labels, 1)
        trg_labels = np.argmax(target_labels, 1)

        assert len(src_labels) == len(fx_src)
        assert len(trg_labels) == len(fx_trg)

        print 'Computing T-SNE.'

        model = TSNE(n_components=2, random_state=0)

        print(plt.style.available)
        plt.style.use('seaborn-paper')

        TSNE_hA = model.fit_transform(np.vstack((fx_src, fx_trg)))
        plt.figure(1,facecolor="white")
        plt.cla()
        plt.scatter(TSNE_hA[:,0], TSNE_hA[:,1], c = np.hstack((src_labels, trg_labels,)),s=10, cmap = mpl.cm.jet)
        plt.savefig('img02/c_%d.eps'%step,format="eps",dpi=1000,bbox_inches="tight")

        plt.figure(2,facecolor="white")
        plt.cla()
        plt.scatter(TSNE_hA[:, 0], TSNE_hA[:, 1], c=np.hstack((np.ones((2000,)), 2 * np.ones((2000,)))), s=10,
                    cmap=mpl.cm.jet)

        plt.savefig('img02/d_%d.eps'%step,format="eps",dpi=1000,bbox_inches="tight")



def main():
    discriminative_loss_param = 0.003 ##0.03 for InstanceBased method, 1e-5 for CenterBased method
    domain_loss_param = 8
    adver_loss_param=0
    param=[discriminative_loss_param, domain_loss_param,adver_loss_param]
    Runer=Train(class_num=10,batch_size=128,iters=200200,learning_rate=0.0001,keep_prob=1,param=param)
    Runer.TrainNet()




def load_mnist(image_dir, split='train'):
    print ('Loading MNIST dataset.')

    image_file = 'train.pkl' if split == 'train' else 'test.pkl'
    image_dir = os.path.join(image_dir, image_file)
    with open(image_dir, 'rb') as f:
        mnist = pickle.load(f)
    images = mnist['X'] / 127.5 - 1
    labels = mnist['y']
    labels=np.squeeze(labels).astype(int)
    return images,labels


def load_svhn(image_dir, split='train'):
    print ('Loading SVHN dataset.')

    image_file = 'train_32x32.mat' if split == 'train' else 'test_32x32.mat'

    image_dir = os.path.join(image_dir, image_file)
    svhn = scipy.io.loadmat(image_dir)
    images = np.transpose(svhn['X'], [3, 0, 1, 2]) / 127.5 - 1
    # ~ images= resize_images(images)
    labels = svhn['y'].reshape(-1)
    labels[np.where(labels == 10)] = 0
    return images, labels


def load_USPS(image_dir,split='train'):
    print('Loading USPS dataset.')
    image_file='USPS_train.pkl' if split=='train' else 'USPS_test.pkl'
    image_dir=os.path.join(image_dir,image_file)
    with open(image_dir, 'rb') as f:
        usps = pickle.load(f)
    images = usps['data']
    images=np.reshape(images,[-1,32,32,1])
    labels = usps['label']
    labels=np.squeeze(labels).astype(int)
    return images,labels



def load_syn(image_dir,split='train'):
    print('load syn dataset')
    image_file='synth_train_32x32.mat' if split=='train' else 'synth_test_32x32.mat'
    image_dir=os.path.join(image_dir,image_file)
    syn = scipy.io.loadmat(image_dir)
    images = np.transpose(syn['X'], [3, 0, 1, 2]) / 127.5 - 1
    labels = syn['y'].reshape(-1)
    return images,labels


def load_mnistm(image_dir,split='train'):
    print('Loading mnistm dataset.')
    image_file='mnistm_train.pkl' if split=='train' else 'mnistm_test.pkl'
    image_dir=os.path.join(image_dir,image_file)
    with open(image_dir, 'rb') as f:
        mnistm = pickle.load(f)
    images = mnistm['data']

    labels = mnistm['label']
    labels=np.squeeze(labels).astype(int)
    return images,labels

def one_hot(x,n):
	if type(x) == list:
		x = np.array(x)
	x = x.flatten()
	o_h = np.zeros((len(x),n))
	o_h[np.arange(len(x)),x] = 1
	return o_h


if __name__=="__main__":

    main()