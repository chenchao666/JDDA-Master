# JDDA-Master
Joint Domain Alignment and Discriminative Feature Learning for Unsupervised Deep Domain Adaptation
* This repository contains code for our paper **Joint Domain Alignment and Discriminative Feature Learning for Unsupervised Deep Domain Adaptation** [Download Paper](https://arxiv.org/abs/1808.09347)

* Another qualified repository completed by the co-author of **JDDA** can be seen [JDDA repository](https://github.com/A-bone1/JDDA)

* Our code contains not only our proposed **JDDA**, but also other famous deep domain adaptation methods **DDC(MMD)**, **DeepCoral**, **DAN(KMMD)** and **CMD**(Central Moment Discrepancy), **LogCoral**, ect.

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

# trainLenet.py
