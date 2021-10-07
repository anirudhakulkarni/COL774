# imports
import collections
from os import supports_effective_ids
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from cvxopt import solvers
from cvxopt import matrix
from libsvm.svmutil import *
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import csv
from PIL import Image
# parameters
digit1=1
digit2=2
C=1
gamma=0.05
threashold=1e-3


def get_accuracy(Y,pred):
    accu=0
    for i in range(len(Y)):
        if Y[i]==pred[i]:
            accu+=1
    accu/=len(Y)
    return accu

def get_d_X_Y(X_train,Y_train, digit1,digit2):
    d_X=[]
    d_Y=[]
    for i in range(len(Y_train)):
        if Y_train[i]==digit1:
            d_X.append(X_train[i])
            d_Y.append(1.0)
        elif Y_train[i]==digit2:
            d_X.append(X_train[i])
            d_Y.append(-1.0)
    d_X=np.array(d_X)
    d_Y=np.array(d_Y)
    return d_X,d_Y

def solve_cvxopt(P,q,G,h,A,b):
    sol=solvers.qp(P,q,G,h,A,b)
    return sol['x']

def draw_confusion(conf,label="linear"):
    plt.figure()
    plt.imshow(conf)
    plt.title("Confusion Matrix"+label)
    plt.colorbar()
    my_xticks = [i for i in range(len(conf))]
    plt.xticks(my_xticks, my_xticks)
    my_yticks = [i for i in range(len(conf))]
    plt.yticks(my_yticks, my_yticks)
    plt.set_cmap("Greens")
    plt.ylabel("True labels")
    plt.xlabel("Predicted label")
    # add points on the axis
    for i in range(len(conf)):
        for j in range(len(conf)):
            plt.text(j,i,str(conf[i,j]),ha="center",va="center",color="black")
    plt.savefig("confusion_matrix"+label+".png")
    plt.show()

def predict(W,b,X_D,Y_D):
    Y_pred=[]
    for i in range(len(X_D)):
        temp=np.dot(X_D[i],W)+b
        if temp>0:
            Y_pred.append(1)
        else:
            Y_pred.append(-1)
    Y_pred=np.array(Y_pred)
    correct=0
    for i in range(len(Y_D)):
        if Y_D[i]==Y_pred[i]:
            correct+=1
    correct=correct/len(Y_D)
    return correct,Y_pred
###########################################################################
# functions for part 1

def get_parameters(d_X,d_Y,C,gamma,kernel):
    m,n=d_X.shape
    A=matrix(d_Y,(1,m))
    b=matrix(0.0)
    h=matrix(np.hstack((np.zeros(m),np.ones(m)*C)))
    G=matrix(np.vstack((-1.0*np.eye(m),np.eye(m))))
    q=matrix(-1.0*np.ones(m))
    if kernel=="linear":
        d_Y=np.array(d_Y)
        temp=d_Y*d_X
        P=temp@temp.T
        P=matrix(P)

    else:
        d_X_2=np.sum(np.multiply(d_X,d_X),axis=1,keepdims=True)
        kernel=d_X_2-2*np.matmul(d_X,d_X.T)+d_X_2.T
        kernel=np.power(np.exp(-gamma),kernel)
        P=matrix(d_Y*d_Y.T*kernel)
    return P,q,G,h,A,b
def multiclass_libsvm_predict(X_train,Y_train,X_test,Y_test,q):
    model = svm_load_model('model.model')
    print("model loaded")
    pred_lbl,pred_acc_train,pred_val=svm_predict(Y_train,X_train,model)
    print(pred_acc_train)
    # plot confusion matrix of train set
    conf_train=confusion_matrix(Y_train,pred_lbl)
    draw_confusion(conf_train,"libsvm-multiclass-train")
    # predict on test data
    pred_lbl_test,pred_acc_test,pred_val=svm_predict(Y_test,X_test,model)
    print(pred_acc_test)
    conf_test=confusion_matrix(Y_test,pred_lbl_test)
    draw_confusion(conf_test,"libsvm-multiclass-test")
    # save missclassified images
    miss_train=[]
    true_train=[]
    false_train=[]
    for i in range(len(Y_train)):
        if Y_train[i]!=pred_lbl[i]:
            miss_train.append(i)
            true_train.append(Y_train[i])
            false_train.append(pred_lbl[i])
    miss_train=np.array(miss_train)
    # save true false labels as csv
    true_train=np.array(true_train)
    false_train=np.array(false_train)

    with open("./multi-libsvm-miss/missclassified_train.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(zip(miss_train,true_train,false_train))

    # save missclassified images
    miss_test=[]
    true_test=[]
    false_test=[]
    for i in range(len(Y_test)):
        if Y_test[i]!=pred_lbl_test[i]:
            miss_test.append(i)
            true_test.append(Y_test[i])
            false_test.append(pred_lbl_test[i])
    miss_test=np.array(miss_test)
    # save true false labels as csv
    true_test=np.array(true_test)
    false_test=np.array(false_test)
    with open("./multi-libsvm-miss/missclassified_test.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(zip(miss_test,true_test,false_test))
    save_images()
def save_images():
    # read 0th column from csv file
    df = pd.read_csv('./multi-libsvm-miss/missclassified_train.csv', usecols=[0],header=None)
    data=pd.read_csv('./dataset/train.csv',header=None)
    # select all but last column
    data=data.iloc[:,:-1]
    data=np.array(data)
    df=np.array(df)
    for i in range(len(df)):
        print(df[i])
        img=Image.fromarray(np.array(data[df[i][0]]).reshape(28,28).astype(np.uint8))
        img.convert('RGB').save("./multi-libsvm-miss/miss_train_"+str(i)+".jpeg")

    # read 0th column from csv file
    df = pd.read_csv('./multi-libsvm-miss/missclassified_test.csv', usecols=[0],header=None)
    data=pd.read_csv('./dataset/test.csv',header=None)
    # select all but last column
    data=data.iloc[:,:-1]
    data=np.array(data)
    df=np.array(df)
    for i in range(len(df)):
        print(df[i])
        img=Image.fromarray(np.array(data[df[i][0]]).reshape(28,28).astype(np.uint8))
        img.convert('RGB').save("./multi-libsvm-miss/miss_test_"+str(i)+".jpeg")


def multiclass_libsvm_v2(X_train,Y_train,X_test,Y_test,q):
    train_time=0
    problem=svm_problem(Y_train,X_train)
    # set parameters
    param=svm_parameter('-s 0 -t 2 -g '+str(gamma)+' -c '+str(C))
    # train model
    start=time.time()
    model=svm_train(problem,param)
    train_time+=time.time()-start
    # predict on train data
    print("Training time:", train_time)
    # save model
    svm_save_model("model.model",model)
    multiclass_libsvm_predict(X_train,Y_train,X_test,Y_test,q)
def multiclass_cvxopt_predict(X_train,Y_train,X_test,Y_test,c):
    X_train,Y_train=read_data(train_path)
    X_test,Y_test=read_data(test_path)
    # default dictionary
    final_pred_train=[collections.defaultdict(int) for i in range(len(Y_train))]
    final_pred_test=[collections.defaultdict(int) for i in range(len(Y_test))]
    train_time=0
    for digit1 in range(10):
        for digit2 in range(digit1+1,10):
            # get data
            print("digit1:",digit1,"digit2:",digit2)
            d_X_train,d_Y_train=get_d_X_Y(X_train,Y_train,digit1,digit2)
            d_Y_train=d_Y_train.reshape(len(d_Y_train),1)
            gaussian_svm=SVM("gaussian",C,gamma)
            gaussian_svm.train(d_X_train,d_Y_train)
            train_time+=gaussian_svm.train_time
            print("Total training time so far:",train_time)

            d_Y_train=d_Y_train.reshape(len(d_Y_train))
            Y_pred_train=gaussian_svm.predict(X_train)
            Y_pred_test=gaussian_svm.predict(X_test)
            # update final_pred_train 
            for i in range(len(X_train)):
                if Y_pred_train[i]==1:
                    final_pred_train[i][digit1]+=1
                else:
                    final_pred_train[i][digit2]+=1
            for i in range(len(X_test)):
                if Y_pred_test[i]==1:
                    final_pred_test[i][digit1]+=1
                else:
                    final_pred_test[i][digit2]+=1
            print(final_pred_train[1])
    final_pred_test_lbl=[]
    final_pred_train_lbl=[]
    for i in range(len(final_pred_train)):
        final_pred_train_lbl.append(max(final_pred_train[i],key=final_pred_train[i].get))
    for i in range(len(final_pred_test)):
        final_pred_test_lbl.append(max(final_pred_test[i],key=final_pred_test[i].get))
    # get accuracy
    accu_train=0
    for i in range(len(Y_train)):
        if Y_train[i]==final_pred_train_lbl[i]:
            accu_train+=1
    accu_train/=len(Y_train)
    print("Multiclass LIBSVM Training accuracy: "+str(accu_train))
    accu_test=0
    for i in range(len(Y_test)):
        if Y_test[i]==final_pred_test_lbl[i]:
            accu_test+=1
        else:
            print("Missclassified",Y_test[i],"as",final_pred_test_lbl[i],"datapoint:",i)
    accu_test/=len(Y_test)
    print("Multiclass LIBSVM Test accuracy: "+str(accu_test))
    # plot confusion matrix
    conf_mat=confusion_matrix(Y_train,final_pred_train_lbl)
    draw_confusion(conf_mat,"cvxopt-multi-train")
    conf_mat=confusion_matrix(Y_test,final_pred_test_lbl)
    draw_confusion(conf_mat,"cvxopt-multi-test")
    
def multiclass_libsvm(X_train,Y_train,X_test,Y_test,part="c"):
    # read data
    X_train,Y_train=read_data(train_path)
    X_test,Y_test=read_data(test_path)
    # default dictionary
    final_pred_train=[collections.defaultdict(int) for i in range(len(Y_train))]
    final_pred_test=[collections.defaultdict(int) for i in range(len(Y_test))]
    train_time=0
    for digit1 in range(10):
        for digit2 in range(digit1+1,10):
            cnt+=1
            print("starting: ",digit1,digit2)
            # get subset of data relevant to digit d and return X and Y
            d_X_train,d_Y_train=get_d_X_Y(X_train,Y_train,digit1,digit2)
            d_X_test,d_Y_test=get_d_X_Y(X_test,Y_test,digit1,digit2)
            # model problem
            problem=svm_problem(d_Y_train,d_X_train)
            # set parameters
            param=svm_parameter('-s 0 -t 0 -g '+str(gamma)+' -c '+str(C))
            # train model
            start=time.time()
            model=svm_train(problem,param)
            train_time+=time.time()-start
            # predict on train data
            pred_lbl,pred_acc_train,pred_val=svm_predict(d_Y_train,d_X_train,model)
            j=0
            for i in range(len(Y_train)):
                if Y_train[i]==digit1:
                    if pred_lbl[j]==1:
                        final_pred_train[i][digit1]+=1
                    else:
                        final_pred_train[i][digit2]+=1
                    j+=1
                elif Y_train[i]==digit2:
                    if pred_lbl[j]==-1:
                        final_pred_train[i][digit2]+=1
                    else:
                        final_pred_train[i][digit1]+=1
                    j+=1
            # predict on test data
            pred_lbl,pred_acc_test,pred_val=svm_predict(d_Y_test,d_X_test,model)
            j=0
            for i in range(len(Y_test)):
                if Y_test[i]==digit1:
                    if pred_lbl[j]==1:
                        final_pred_test[i][digit1]+=1
                    else:
                        final_pred_test[i][digit2] +=1
                    j+=1
                elif Y_test[i]==digit2:
                    if pred_lbl[j]==-1:
                        final_pred_test[i][digit2]+=1
                    else:
                        final_pred_test[i][digit1] +=1
                    j+=1
                

    final_pred_test_lbl=[]
    final_pred_train_lbl=[]
    for i in range(len(final_pred_train)):
        final_pred_train_lbl.append(max(final_pred_train[i],key=final_pred_train[i].get))
    for i in range(len(final_pred_test)):
        final_pred_test_lbl.append(max(final_pred_test[i],key=final_pred_test[i].get))
    # get accuracy
    accu_train=0
    for i in range(len(Y_train)):
        if Y_train[i]==final_pred_train_lbl[i]:
            accu_train+=1
    accu_train/=len(Y_train)
    print("Multiclass CVXOPT Training accuracy: "+str(accu_train))
    accu_test=0
    for i in range(len(Y_test)):
        if Y_test[i]==final_pred_test_lbl[i]:
            accu_test+=1
        else:
            print("Missclassified",Y_test[i],"as",final_pred_test_lbl[i],"datapoint:",i)
    accu_test/=len(Y_test)
    print("Multiclass CVXOPT Test accuracy: "+str(accu_test))
    if part=="c":
        # confusion matrix
        conf_matrix_train=confusion_matrix(Y_train,final_pred_train_lbl)
        conf_matrix_test=confusion_matrix(Y_test,final_pred_test_lbl)
        # write results to "svm_libsvm_multiclass.txt"
        with open("svm_libsvm_multiclass.txt", "w") as f:
            f.write("Train time: "+str(train_time)+"\n")
            f.write("Multiclass CVXOPT Training accuracy: "+str(accu_train)+"\n")
            f.write("Multiclass CVXOPT Test accuracy: "+str(accu_test)+"\n")
            f.write("Multiclass CVXOPT Training confusion matrix: \n")
            f.write(str(conf_matrix_train)+"\n")
            f.write("Multiclass CVXOPT Test confusion matrix: \n")
            f.write(str(conf_matrix_test)+"\n")
        # plot confusion matrix
        draw_confusion(conf_matrix_train,"_libsvm_multiclass_train")
        draw_confusion(conf_matrix_test,"_libsvm_multiclass_test")

class SVM:
    '''
    parameters:
    C: regularization parameter
    gamma: kernel parameter
    kernel: kernel type

    '''
    def __init__(self,kernel,C,gamma):
        self.kernel=kernel
        self.C=C
        self.gamma=gamma
        self.train_time=0
    
    def train(self,X,Y):
        # get parameters
        m,n=X.shape
        Y=Y.reshape(len(Y),1)
        P,q,G,h,A,b = get_parameters(X,Y,self.C,self.gamma,self.kernel)
        # solve QP
        print(P.size,q.size,G.size,h.size,A.size,b.size)

        start=time.time()
        alpha=np.ravel(solve_cvxopt(P,q,G,h,A,b)).reshape(m,1)
        end=time.time()
        # get support vectors
        print("Train time:",end-start)
        self.train_time+=end-start
        self.supp_flag=(alpha>threashold).ravel()
        self.supp_indices=np.arange(len(alpha))[self.supp_flag]
        self.alpha=alpha[self.supp_flag]
        self.supp_vec=X[self.supp_flag]
        self.supp_vec_y=Y[self.supp_flag]
        if self.kernel=="linear":
            self.w=np.sum(self.alpha*self.supp_vec*self.supp_vec_y,axis=0,keepdims=True).T
            temp0=np.dot(X,self.w)
            minsofar=1e10
            maxsofar=-1e10
            for i in range(len(temp0)):
                if Y[i]==1:
                    if temp0[i]<minsofar:
                        minsofar=temp0[i]
                else:
                    if temp0[i]>maxsofar:
                        maxsofar=temp0[i]
            self.b=-(minsofar+maxsofar)/2
        print(len(self.supp_vec_y))

    def predict(self,X):
        print("Predicting ...")
        if self.kernel=="linear":
            return np.sign(np.dot(X,self.w)+self.b)
        m=len(X)
        pred=np.zeros(m)
        self.alpha=self.alpha.ravel()
        self.supp_vec_y=self.supp_vec_y.ravel()
        # print(self.alpha.shape,self.supp_vec.shape,self.supp_vec_y.shape)
        # print(self.alpha)
        # print(self.supp_vec)
        # print(self.supp_vec_y)
        # return
        # tick=time.time()
        # saved_pr=np.dot(self.alpha,np.diag(self.supp_vec_y))
        # for i in range(m):
        #     print(i)
        #     pred[i]=np.sum(np.dot(saved_pr,np.diag(np.exp(-self.gamma*np.linalg.norm(X[i]-self.supp_vec,axis=1)**2))))
        # tock=time.time()
        # print("1Predict time:",tock-tick)
        # works best
        tick=time.time()
        saved_pr=self.alpha*self.supp_vec_y
        for i in range(m):
            pred[i]=np.sum(saved_pr*np.diag(np.exp(-self.gamma*np.linalg.norm(X[i]-self.supp_vec,axis=1)**2)))
        tock=time.time()
        print("Predict time:",tock-tick)
        return np.sign(pred)
    


def read_data(filename='./dataset/train.csv'):
    dataset = pd.read_csv(filename, header=None)
    X=dataset.iloc[:,:-1]
    X=X.values
    X=X/255
    Y=dataset.iloc[:,-1]
    Y=Y.values
    return X,Y


def cross_validation(X_train,Y_train):
    C_set=[10,5,1,1e-3,1e-5]
    K_fold=5
    # divide data into K_fold folds
    X_train_folds=[]
    Y_train_folds=[]
    for i in range(K_fold):
        X_train_folds.append([])
        Y_train_folds.append([])
    for i in range(len(X_train)):
        X_train_folds[i%K_fold].append(X_train[i])
        Y_train_folds[i%K_fold].append(Y_train[i])
    # cross validation
    acc_set=[]
    for C in C_set:
        # get accuracy
        acc=0
        print("C:",C,"\n---------------------")
        for i in range(K_fold):
            print("Fold:",str(i+1))
            # get subset of data relevant to fold i
            X_train_subset=[]
            Y_train_subset=[]
            for j in range(K_fold):
                if i!=j:
                    X_train_subset+=X_train_folds[j]
                    Y_train_subset+=Y_train_folds[j]
            X_test_subset=X_train_folds[i]
            Y_test_subset=Y_train_folds[i]
            train_time=0
            problem=svm_problem(Y_train_subset,X_train_subset)
            # set parameters
            param=svm_parameter('-s 0 -q -t 2 -g '+str(gamma)+' -c '+str(C))
            model=svm_train(problem,param)
            # predict on test data
            pred_lbl_test,pred_acc_test,pred_val=svm_predict(Y_test_subset,X_test_subset,model)
            # print("Validation accuracy:", pred_acc_test[0])    
            acc+=pred_acc_test[0]      
        acc_set.append(acc/K_fold)
    
    print("Accuracy set:",acc_set)



if __name__ == '__main__':
    # get command line arguments
    train_path=sys.argv[1]
    test_path=sys.argv[2]
    if len(sys.argv)>4:
        class_num=int(sys.argv[3])
        part_num=sys.argv[4]
    else:
        class_num=1
        part_num=sys.argv[3]
    # read data
    print("Reading data ...")
    X_train,Y_train=read_data(train_path)
    X_test,Y_test=read_data(test_path)    
    # binary classification
    if class_num==0:
        # get subset of data relevant to digit d and return X and Y
        d_X_train,d_Y_train=get_d_X_Y(X_train,Y_train,digit1,digit2)
        d_X_test,d_Y_test=get_d_X_Y(X_test,Y_test,digit1,digit2)
        print("Training SVM ...")
        if part_num=='a':
            d_Y_train=d_Y_train.reshape(len(d_Y_train),1)
            linear_svm=SVM("linear",C,gamma)
            linear_svm.train(d_X_train,d_Y_train)

            pred=linear_svm.predict(d_X_train)
            accu=get_accuracy(d_Y_train,pred)
            print(accu)
            pred=linear_svm.predict(d_X_test)
            accu=get_accuracy(d_Y_test,pred)
            print(accu)
            print(linear_svm.b)
            # tasks:
            # 1. obtain support vectors and save to file
            # 2. print bias
            # 3. print weights
            # 4. print validation accuracy
            # 5. print test accuracy
            # 6. print time required
        elif part_num=='b':
            d_Y_train=d_Y_train.reshape(len(d_Y_train),1)
            gaussian_svm=SVM("gaussian",C,gamma)
            gaussian_svm.train(d_X_train,d_Y_train)
            d_Y_train=d_Y_train.reshape(len(d_Y_train))
            d_Y_pred=gaussian_svm.predict(d_X_train)
            print(d_Y_pred)
            print(d_Y_train)
            accu=get_accuracy(d_Y_train,d_Y_pred)
            print(accu)
            d_Y_pred=gaussian_svm.predict(d_X_test)
            accu=get_accuracy(d_Y_test,d_Y_pred)
            print(accu)
            # tasks:
            # 1. obtain support vectors and save to file
            # 2. print validation accuracy
            # 3. print test accuracy
            # 4. print time required
        elif part_num=='c':
            # libsvm_linear_gaussian(train_path,test_path)
            problem=svm_problem(d_Y_train,d_X_train)
            # linear kernel
            linear_param=svm_parameter('-s 0 -t 0 -c '+str(C))
            start_lin=time.time()
            linear_model=svm_train(problem,linear_param)
            end_lin=time.time()
            # predict on train data
            linear_pred_lbl_train,linear_pred_acc_train,linear_pred_val=svm_predict(d_Y_train,d_X_train,linear_model)
            # predict on test data
            linear_pred_lbl_test,linear_pred_acc_test,linear_pred_val=svm_predict(d_Y_test,d_X_test,linear_model)
            # gaussian kernel
            gaussian_param=svm_parameter('-s 0 -t 2 -g '+str(gamma)+' -c '+str(C))
            start_gauss=time.time()
            gaussian_model=svm_train(problem,gaussian_param)
            end_gauss=time.time()
            # predict on train data
            gaussian_pred_lbl_train,gaussian_pred_acc_train,gaussian_pred_val=svm_predict(d_Y_train,d_X_train,gaussian_model)
            # predict on test data
            gaussian_pred_lbl_test,gaussian_pred_acc_test,gaussian_pred_val=svm_predict(d_Y_test,d_X_test,gaussian_model)
            # write results to "svm_libsvm_linear_gaussian.txt"
            with open("svm_libsvm_linear_gaussian.txt", "w") as f:
                f.write("Linear LIBSVM Training time: "+str(end_lin-start_lin)+"\n")
                f.write("Gaussian LIBSVM Training time: "+str(end_gauss-start_gauss)+"\n")
                f.write("Linear LIBSVM train accuraccy: "+str(linear_pred_acc_train[0])+"\n") 
                f.write("Linear LIBSVM test accuraccy: "+str(linear_pred_acc_test[0])+"\n")
                f.write("Gaussian LIBSVM train accuraccy: "+str(gaussian_pred_acc_train[0])+"\n")
                f.write("Gaussian LIBSVM test accuraccy: "+str(gaussian_pred_acc_test[0])+"\n")
            # save confusion matrix plots
            conf_matrix_train_lin=confusion_matrix(d_Y_train,linear_pred_lbl_train)
            conf_matrix_test_lin=confusion_matrix(d_Y_test,linear_pred_lbl_test)
            draw_confusion(conf_matrix_train_lin,"_libsvm_linear_train")
            draw_confusion(conf_matrix_test_lin,"_libsvm_linear_test")
            conf_matrix_train_gauss=confusion_matrix(d_Y_train,gaussian_pred_lbl_train)
            conf_matrix_test_gauss=confusion_matrix(d_Y_test,gaussian_pred_lbl_test)
            draw_confusion(conf_matrix_train_gauss,"_libsvm_gaussian_train")
            draw_confusion(conf_matrix_test_gauss,"_libsvm_gaussian_test")

            # tasks:
            # 1. obtain support vectors and save to file
            # 2. print bias
            # 3. print weights
            # 4. print validation accuracy
            # 5. print test accuracy
            # 6. print time required
        else:
            print('wrong part number')
            exit()

    elif class_num==1:

        if part_num=='a':
            multiclass_cvxopt_predict(X_train,Y_train,X_test,Y_test,"c")
            # multiclass_cvxopt(train_path,test_path)
        elif part_num=='b':
            # to train the model and save the parameters
            # multiclass_libsvm_v2(train_path,test_path,"c")
            # to predict the labels and save the results
            multiclass_libsvm_predict(X_train,Y_train,X_test,Y_test,"c")
        elif part_num=='c':
            multiclass_libsvm_predict(X_train,Y_train,X_test,Y_test,"c")
        elif part_num=='d':
            cross_validation(X_train,Y_train)

        else:
            print('wrong part number')
            exit()
    else:


        print("class_num should be 0 or 1")
        exit()

