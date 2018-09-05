# import cv2
import os
import matplotlib.pyplot as plt
from numpy import *
import pickle
import random
from sklearn.decomposition import PCA


class DataLoader:
    def __init__(self, Dataset,source="Amazon",target="Dslr"):
        self.Dataset=Dataset
        self.source_name=source
        self.target_name=target




    def Read(self):
        self.IMAGE=zeros((4110,227,227,3))
        self.IMAGE=uint8(self.IMAGE)
        k=0
        m = 0
        self.NumList=zeros(93)
        if self.Dataset=="office31":
            list=os.listdir("./Dataset/office31/")
            list.sort()
            for filename in list:
                list1=os.listdir("./Dataset/office31/"+filename+"/"+"images/")
                list1.sort()
                for classes in list1:
                    list2=os.listdir("./Dataset/office31/"+filename+"/"+"images/"+classes)
                    list2.sort()
                    self.NumList[m]=len(list2)
                    m+=1
                    for imgname in list2:
                        img=cv2.imread("./Dataset/office31/"+filename+"/"+"images/"+classes+"/"+imgname)
                        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                        img=cv2.resize(img,(227,227),interpolation=cv2.INTER_CUBIC)
                        self.IMAGE[k,:,:,:]=img
                        k+=1


            self.Construct()



    def Construct(self):
        print self.NumList
        print sum(self.NumList)
        amazon_list=self.NumList[:31]
        dslr_list=self.NumList[31:62]
        webcam_list=self.NumList[62:]
        print sum(amazon_list)
        print sum(dslr_list)
        print sum(webcam_list)
        Amazon={}
        Dslr={}
        Webcam={}
        Amazon["Data"]=self.IMAGE[:sum(amazon_list),:,:,:]
        Dslr["Data"]=self.IMAGE[sum(amazon_list):sum(amazon_list)+sum(dslr_list),:,:,:]
        Webcam["Data"]=self.IMAGE[sum(amazon_list)+sum(dslr_list):sum(amazon_list)+sum(dslr_list)+sum(webcam_list),:,:,:]
        Amazon["Label"]=self.list2LabelMatrix(amazon_list)
        Dslr["Label"]=self.list2LabelMatrix(dslr_list)
        Webcam["Label"]=self.list2LabelMatrix(webcam_list)
        output1=open("Amazon.pkl","wb")
        pickle.dump(Amazon,output1)
        output2 = open("Dslr.pkl", "wb")
        pickle.dump(Dslr,output2)
        output3 = open("Webcam.pkl", "wb")
        pickle.dump(Webcam,output3)




    def list2LabelMatrix(self,Num_list):
        num=sum(Num_list)
        Label=zeros((num,31))
        List=Num_list.tolist()
        List.insert(0,0)
        List=cumsum(List)
        for i in range (31):
            Label[List[i]:List[i+1],i]=1
        return Label








##############################################################################


    def LoadSource(self):
        filename=self.source_name+".pkl"
        f=open(filename,"r")
        Source=pickle.load(f)
        SourceData=Source.get("Data")
        SourceLabel=Source.get("Label")
        # SourceData=SourceData/255.0
        f.close()
        return self.shuffle(SourceData, SourceLabel)



    def LoadTarget(self):
        filename=self.target_name+".pkl"
        f=open(filename,"r")
        Target=pickle.load(f)
        TargetData=Target.get("Data")
        TargetLabel=Target.get("Label")
        # TargetData=TargetData/255.0
        TargetData,TargetLabel=self.shuffle(TargetData,TargetLabel)
        return TargetData, TargetData, TargetLabel




    def shuffle(self,Data,Label):
        ind=range(Data.shape[0])
        random.shuffle(ind)
        Data=Data[ind,:,:,:]
        Label=Label[ind,:]
        return Data, Label

###################################################################

def CalLapMatrix(Data,alpha,K):
    dist=Cal_pairwise_dist(Data)
    W=exp(-dot(dist,alpha))
    W=get_Ksparse(W,K)
    D=get_D(W)
    LapMatrix=D-W
    return W, LapMatrix



def Cal_pairwise_dist(Data):
    DataVector=ImgVectorize(Data)
    pca=PCA(n_components=5)
    DataVector=pca.fit_transform(DataVector)
    norm = lambda x: sum(square(x), 1)
    dist = transpose(norm(expand_dims(DataVector, 2) - transpose(DataVector)))
    # dist=zeros((DataVector.shape[0],DataVector.shape[0]))
    # for i in range(DataVector.shape[0]):
    #     for j in range(DataVector.shape[0]):
    #         dist[i,j]=linalg.norm(DataVector[i,:]-DataVector[j,:])
    return dist


def ImgVectorize(Data):
    datasize=Data.shape[1]*Data.shape[2]*Data.shape[3]
    DataVector=zeros((Data.shape[0],datasize))
    for i in range(Data.shape[0]):
        img=Data[i,:]
        # img=cv2.resize(img,(32,32),interpolation=cv2.INTER_CUBIC)
        DataVector[i,:]=img.flatten()
    # DataVector /= 255.0
    return DataVector



def ImgVectorize_CNN(Data):
    vgg_model=resnet50.ResNet50(weights="imagenet")
    processed_image=resnet50.preprocess_input(Data.copy())
    predictions=vgg_model.predict(processed_image)
    return predictions



def get_Ksparse(Data,K):
    for i in range(Data.shape[0]):
        X=Data[i,:]
        X=list(X)
        X.sort(reverse=True)
        ind=Data[i,:] > X[K]
        ind=ind.astype("float")
        Data[i,:]=ind
        # Data[i,:]=Data[i,:] * ind
    return Data



def get_D(Data):
    D=zeros(Data.shape)
    for i in range(Data.shape[0]):
        D[i,i]=sum(Data[i,:])
    return D











if __name__=="__main__":
    Data = DataLoader("office31", source="amazon", target="dslr")
    A=Data.Read()

    # f=open("Amazon.pkl","r")
    # Amazon=pickle.load(f)
    # Data=Amazon.get("Data")
    # Label=Amazon.get("Label")
    # f.close()
    # for i in range(100):
    #     plt.imshow(Data[i,:,:,:])
    #     plt.show()


















