# JDDA-Master
Joint Domain Alignment and Discriminative Feature Learning for Unsupervised Deep Domain Adaptation
* This repository contains code for our paper **Joint Domain Alignment and Discriminative Feature Learning for Unsupervised Deep Domain Adaptation** [Download Paper](https://arxiv.org/abs/1808.09347)

* Another qualified repository completed by the co-author of **JDDA** can be seen [JDDA repository](https://github.com/A-bone1/JDDA)

* Our code contains not only our proposed **JDDA**, but also other famous deep domain adaptation methods **DDC(MMD)**, **DeepCoral**, **DAN(KMMD)** and **CMD**(Central Moment Discrepancy), **LogCoral**, ect.

* If you have any question about our [paper](https://arxiv.org/abs/1808.09347) or code, please don't hesitate to contact with me **ahucomputer@126.com**, we will update our repository accordingly

# Movition of our proposal
* Most of existing work only concentrates on learning shared feature representation by minimizing the distribution discrepancy across different domains. Due to the fact that all the domain alignment approaches can only reduce, but not remove the domain shift, target domain samples distributed near the edge of the clusters, or far from their corresponding class centers are easily to be misclassified by the hyperplane learned from the source domain. To alleviate this issue, we propose to joint domain alignment and discriminative feature learning, which could benefit both domain alignment and final classification. 
The necessity of joint domain alignment and discriminant features learning can be seen below
![image](https://github.com/chenchao666/JDDA-Master/blob/master/img/fig1.jpg)

# Result 
* Our proposed JDDA achieves a state-of-art results among those completing Discrepancy-Based Domain Adaptation methods. 
![image](https://github.com/chenchao666/JDDA-Master/blob/master/img/fig3.jpg)

* The t-sne  visualization, with the incorporation of our proposed discriminative loss, the deep features become better clusterred and more separable
![image](https://github.com/chenchao666/JDDA-Master/blob/master/img/fig4.jpg)


# Run The Code
* This code requires Python 2.7 and implemented in Tensorflow 1.9. You can download all the datasets used in our paper from [Dataset](https://pan.baidu.com/s/1IMUVnpM8Ve6XX37rtv2zJQ#list/path=%2F) and place them in the specified directory, and place them in the specified directory. Run **trainLenet.py** to obtain the results. 

## trainLenet.py
* the Core Code of our proposed **Instance-Based** and **Center-Based** discriminative feature learning can be seen in trainLenet.py
``` python
        ## Instence-Based Discriminative Feature Learning
        ## Xs is the deep features from the source domain with its row-num equals to batchsize and colum-num equals to neural of neurons in the adapted layer
	## self.W is the indicator matrix. self.W[i,j]=1 means i-th and j-th samples are from the same calss, self.W[i,j]=0 
	## means i-th and j-th samples are from difference calsses
    def CalDiscriminativeLoss(self,method):
        if method=="InstanceBased":
            Xs = self.source_model.fc4
            norm = lambda x: tf.reduce_sum(tf.square(x), 1)
            self.F0 = tf.transpose(norm(tf.expand_dims(Xs, 2) - tf.transpose(Xs)))  #calculate pair-wise distance of Xs
            margin0 = 0
            margin1 = 100
            F0=tf.pow(tf.maximum(0.0, self.F0-margin0),2)
            F1=tf.pow(tf.maximum(0.0, margin1-self.F0),2)
            self.intra_loss=tf.reduce_mean(tf.multiply(F0, self.W))
            self.inter_loss=tf.reduce_mean(tf.multiply(F1, 1.0-self.W))
            self.discriminative_loss = (self.intra_loss+self.inter_loss) / (self.BatchSize * self.BatchSize)


        ## Center-Based Discriminative Feature Learning, Note that the center_loss.py should be import 
	## Note that when using the Center-Based Discriminative Loss, the "global class center" should be also update in each iteration by using
	## with tf.control_dependencies([self.centers_update_op]):
        ## self.solver = tf.train.AdamOptimizer(learning_rate=self.LearningRate).minimize(self.loss)
        elif method=="CenterBased":
            Xs=self.source_model.fc4
            labels=tf.argmax(self.source_label,1)
            self.inter_loss, self.intra_loss, self.centers_update_op = get_center_loss(Xs, labels, 0.5, 10)
            self.discriminative_loss = self.intra_loss+ self.inter_loss
            self.discriminative_loss=self.discriminative_loss/(self.ClassNum*self.BatchSize+self.ClassNum*self.ClassNum)
```
