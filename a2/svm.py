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
'''
LIBSVM parameters
         - -s svm_type : set type of SVM (default 0)

            - 0 -- C-SVC        (multi-class classification)
            - 1 -- nu-SVC        (multi-class classification)
            - 2 -- one-class SVM
            - 3 -- epsilon-SVR    (regression)
            - 4 -- nu-SVR        (regression)
        
        - -t kernel_type : set type of kernel function (default 2)
        
            - 0 -- linear: u\'\*v
            - 1 -- polynomial: (gamma\*u\'\*v + coef0)^degree
            - 2 -- radial basis function: exp(-gamma\*|u-v|^2)
            - 3 -- sigmoid: tanh(gamma\*u\'\*v + coef0)
            - 4 -- precomputed kernel (kernel values in training_set_file)
        
        - -d degree : set degree in kernel function (default 3)
        - -g gamma : set gamma in kernel function (default 1/num_features)
        - -r coef0 : set coef0 in kernel function (default 0)
        - -c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
        - -n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
        - -p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
        - -m cachesize : set cache memory size in MB (default 100)
        - -e epsilon : set tolerance of termination criterion (default 0.001)
        - -h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)
        - -b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
        - -wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)
        - -v n: n-fold cross validation mode
        - -q : quiet mode (no outputs)'''
# parameters
digit1=1
digit2=2
C=1
gamma=0.05
threashold=1e-5

# common functions
# Take file path and return X and Y seperately
def read_data(filename='./dataset/train.csv'):
    dataset = pd.read_csv(filename, header=None)
    X=dataset.iloc[:,:-1]
    X=X.values
    X=X/255
    Y=dataset.iloc[:,-1]
    Y=Y.values
    return X,Y

# get subset of data relevant to digit d and return X and Y
def get_d_X_Y(X_train,Y_train, digit1,digit2):
    # return data with label = d
    d_X=X_train[(Y_train==digit1).ravel() | (Y_train==digit2).ravel()]
    d_Y=Y_train[(Y_train==digit1).ravel() | (Y_train==digit2).ravel()]
    d_Y=-1.0*(d_Y==digit1) + 1.0*(d_Y==digit2)
    # d_X=[]
    # d_Y=[]
    # for i in range(len(Y_train)):
    #     if Y_train[i]==digit1:
    #         d_X.append(X_train[i])
    #         d_Y.append(1)
    #     elif Y_train[i]==digit2:
    #         d_X.append(X_train[i])
    #         d_Y.append(-1)
    # d_X=np.array(d_X)
    # d_Y=np.array(d_Y)
    return d_X,d_Y

# solve quadratic optimization with CVXOPT
# 1/2*xtPx+qtx
# Gx<=h
# Ax=b
def solve_cvxopt(P,q,G,h,A,b):
    sol=solvers.qp(P,q,G,h,A,b)
    return sol['x']


# get support vectors
def get_support_vectors(alpha_D,d_X,d_Y):
    scount=0
    support_vectors=[]
    support_vectors_indices=[]
    for i in range(len(alpha_D)):
        if alpha_D[i]>threashold:
            scount+=1
            support_vectors.append(alpha_D[i])
            support_vectors_indices.append(i)
        else:
            support_vectors.append(0)
    support_vectors=np.array(support_vectors)
    # save support vectors to "support_vectors_linear.txt" file
    with open("support_vectors_linear.txt", "w") as f:
        for i in range(len(support_vectors_indices)):
            f.write(str(support_vectors_indices[i])+"\n")
    return support_vectors,scount


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
def get_parameters_linear(d_X,d_Y,C):
    d_Y=d_Y.astype('float')
    A=matrix(d_Y).T
    b=matrix(0.0)
    h=matrix(np.zeros((len(d_X)*2,1)))
    for i in range(len(d_X),2*len(d_X)):
        h[i,0]=C
    G=matrix(np.zeros((len(d_X)*2,len(d_X))))
    for i in range(len(d_X)):
        G[i,i]=-1
        G[i+len(d_X),i]=1
    q=matrix(-np.ones((len(d_X),1)))
    P=matrix(np.zeros((len(d_X),len(d_X))))
    for i in range(len(d_X)):
        for j in range(len(d_X)):
            P[i,j]=np.dot(d_X[i],d_X[j])
            P[i,j]*=d_Y[i]*d_Y[j]
    return P,q,G,h,A,b
def get_w_b(support_vectors,X_D,Y_D):
    W=[]
    for i in range(len(support_vectors)):
        if i==0:
            W=support_vectors[i]*Y_D[i]*X_D[i]
        else:
            W+=support_vectors[i]*Y_D[i]*X_D[i]
    W=np.array(W)
    temp0=np.dot(X_D,W)
    minsofar=1e10
    maxsofar=-1e10
    for i in range(len(temp0)):
        if Y_D[i]==1:
            if temp0[i]<minsofar:
                minsofar=temp0[i]
        else:
            if temp0[i]>maxsofar:
                maxsofar=temp0[i]
    b=-(minsofar+maxsofar)/2
    return W,b
def linear_cvxopt(train_path,test_path):
    # read data
    X_train,Y_train=read_data(train_path)
    X_test,Y_test=read_data(test_path)
    # get subset of data relevant to digit d and return X and Y
    d_X_train,d_Y_train=get_d_X_Y(X_train,Y_train,digit1,digit2)
    d_X_test,d_Y_test=get_d_X_Y(X_test,Y_test,digit1,digit2)
    # get parameters
    P,q,G,h,A,b=get_parameters_linear(d_X_train,d_Y_train,C)
    print("parameter generation complete")
    # solve
    # measure time to solve the problem
    start=time.time()
    alpha=solve_cvxopt(P,q,G,h,A,b)
    end=time.time()
    print("solving time:",end-start)
    # get support vectors
    sv,nSV=get_support_vectors(alpha,d_X_train,d_Y_train)
    # get w and b
    w,b=get_w_b(sv,d_X_train,d_Y_train)
    # predict on train data
    accu_train,d_Y_pred_train=predict(w,b,d_X_train,d_Y_train)
    # predict on test data
    accu_test,d_Y_pred_test=predict(w,b,d_X_test,d_Y_test)
    # get confusion matrix for train data
    conf_train=confusion_matrix(d_Y_train,d_Y_pred_train)
    f1_train=f1_score(d_Y_train,d_Y_pred_train,average=None)
    macro_f1_train=f1_score(d_Y_train,d_Y_pred_train,average='macro')
    # get confusion matrix for test data
    conf_test=confusion_matrix(d_Y_test,d_Y_pred_test)
    f1_test=f1_score(d_Y_test,d_Y_pred_test,average=None)
    macro_f1_test=f1_score(d_Y_test,d_Y_pred_test,average='macro')
    
    # save linear cvxopt results to "svm_linear_cvxopt.txt"
    with open("svm_linear_cvxopt.txt", "w") as f:
        f.write("Training time: "+str(end-start)+"\n")
        f.write("Number of support vectors: "+str(nSV)+"\n")
        f.write("Bias: "+str(b)+"\n")
        f.write("Accuracy on train data: "+str(accu_train)+"\n")
        f.write("F1 score on train data: "+str(f1_train)+"\n")
        f.write("Macro F1 score on train data: "+str(macro_f1_train)+"\n")
        f.write("Confusion matrix on train data:\n")
        for i in range(len(conf_train)):
            for j in range(len(conf_train[0])):
                f.write(str(conf_train[i][j])+" ")
            f.write("\n")
        f.write("Accuracy on test data: "+str(accu_test)+"\n")
        f.write("F1 score on test data: "+str(f1_test)+"\n")
        f.write("Macro F1 score on test data: "+str(macro_f1_test)+"\n")
        f.write("Confusion matrix on test data:\n")
        for i in range(len(conf_test)):
            for j in range(len(conf_test[0])):
                f.write(str(conf_test[i][j])+" ")
            f.write("\n")
    with open("svm_linear_cvxopt_weights.txt", "w") as f:
        f.write("Weights: "+str(w)+"\n")
    # save confusion matrix plots _libsvm_multiclass_train
    draw_confusion(conf_train,"_cvxopt_linear_train")
    draw_confusion(conf_test,"_cvxopt_linear_test")
    
# Gaussian kernel
def get_parameters_gaussian_kernel(d_X,d_Y,C,gamma):
    m,n=d_X.shape
    A=matrix(d_Y,(1,m))
    b=matrix(0.0)
    # h=matrix(np.zeros((len(d_X)*2,1)))
    # for i in range(len(d_X),2*len(d_X)):
    #     h[i,0]=C
    h1=np.zeros(m)
    h2=np.ones(m)*C
    h=matrix(np.hstack((h1,h2)))
    # G=matrix(np.zeros((len(d_X)*2,len(d_X))))
    # for i in range(len(d_X)):
    #     G[i,i]=-1
    #     G[i+len(d_X),i]=1
    G=matrix(np.vstack((-1.0*np.eye(m),np.eye(m))))
    q=matrix(-1.0*np.ones(m))
    # kernel=np.array([d_X,]*len(d_X)).T-np.array([d_X,]*len(d_X))
    # method 1
    # S=np.dot(d_X,d_X.T)
    # S_norm=np.linalg.norm(S)
    d_X_2=np.sum(np.multiply(d_X,d_X),axis=1,keepdims=True)
    kernel=d_X_2-2*np.matmul(d_X,d_X.T)+d_X_2.T
    kernel=np.power(np.exp(-gamma),kernel)
    # kernel=np.asmatrix(np.zeros((m,m),dtype='float'))
    # for i in range(m):
    #     if i%100==0:
    #         print(i)
    #     for j in range(m):
    #         # kernel[i,j]=np.exp(-gamma*(S[i][i]+S[j][j]-2*S[i][j]))
    #         kernel[i,j]=np.exp(-gamma*np.linalg.norm(d_X[i]-d_X[j])**2)
    # method 2
    # from scipy.spatial.distance import pdist, squareform
    # pairwise_dists = squareform(pdist(d_X, 'euclidean'))
    # kernel = np.exp(-gamma*pairwise_dists**2 )

    print("Starting P")
    # P=matrix(np.zeros((len(d_X),len(d_X))))
    P=matrix(d_Y*d_Y.T*kernel)
    # for i in range(len(d_X)):
    #     for j in range(len(d_X)):
    #         P[i,j]=kernel[i,j]
    #         P[i,j]*=d_Y[i]*d_Y[j]
    return P,q,G,h,A,b
def gaussian_cvxopt(train_path,test_path):
    # read data
    train_data=np.genfromtxt(train_path,delimiter=',')
    X_train=train_data[:,0:784]/255
    Y_train=train_data[:,784].reshape(len(train_data),1)
    test_data=np.genfromtxt(test_path,delimiter=',')
    X_test=test_data[:,0:784]/255
    Y_test=test_data[:,784].reshape(len(test_data),1)
    
    # get subset of data relevant to digit d and return X and Y
    d_X_train,d_Y_train=get_d_X_Y(X_train,Y_train,digit1,digit2)
    d_X_test,d_Y_test=get_d_X_Y(X_test,Y_test,digit1,digit2)
    # get parameters
    d_Y_train=d_Y_train.reshape(len(d_Y_train),1)
    P,q,G,h,A,b=get_parameters_gaussian_kernel(d_X_train,d_Y_train,C,gamma)
    print("parameter generation complete")
    # solve
    # measure time to solve the problem
    m,n=d_X_train.shape
    start=time.time()
    alpha=np.ravel(solve_cvxopt(P,q,G,h,A,b)).reshape(m,1)
    end=time.time()
    supp_flag=(alpha>threashold).ravel()
    supp_indices=np.arange(len(alpha))[supp_flag]
    alpha=alpha[supp_flag]
    supp_vec=d_X_train[supp_flag]
    
    supp_vec_y=d_Y_train[supp_flag]
    m=len(d_X_test)
    pred=np.zeros(m)
    for i in range(m):
        temp=0
        s=0
        for alpha_i, supp_vec_i, supp_vec_y_i in zip(alpha, supp_vec,supp_vec_y):
            s+=alpha_i*supp_vec_y_i*np.exp(-gamma*np.linalg.norm(d_X_test[i]-supp_vec_i)**2)
    
        pred[i]=s
    # calculate accuracy
    acc=0
    pred=np.sign(pred)
    for i in range(len(pred)):
        if d_Y_test[i]==pred[i]:
            acc+=1
    print(acc/float(len(d_Y_test)))
    return
    print("solving time:",end-start)
    nSV=0
    d_X_train_mul=np.sum(np.multiply(d_X_train,d_X_train),axis=1)
    d_X_test_mul=np.sum(np.multiply(d_X_test,d_X_test),axis=1)
    d_X_train_d_X_test=np.dot(d_X_train,d_X_test.T)
    alpha_x=np.asmatrix(np.zeros((len(alpha),1),dtype='float'))
    for i in range(len(alpha)):
        if alpha[i]>threashold:
            alpha_x[i,0]=alpha[i]*d_Y_train[i]
            nSV+=1
    support_vector_indx=np.arange(len(alpha))[alpha>threashold]
    b=0
    for i in support_vector_indx:
        b+=d_Y_train[i]-np.sum(np.multiply(alpha_x,np.exp(-gamma*np.sum(np.multiply(d_X_train-d_X_train[i,:],d_X_train-d_X_train[i,:]),axis=1))))
    b=b/float(nSV)
    d_Y_pred_test=np.zeros((len(d_X_test),1),dtype=int)
    print("starting predictions")
    for i in range(len(d_X_test)):
        d_Y_pred_test[i,0]=-np.sign(np.sum(np.multiply(alpha_x,np.exp(-gamma*(d_X_train_mul-2*d_X_train_d_X_test[:,i]+d_X_test_mul[i])))))+b
    acc=0
    print(d_Y_pred_test)
    for i in range(len(d_Y_test)):
        if d_Y_test[i]==d_Y_pred_test[i,0]:
            acc+=1
    print(acc/float(len(d_Y_test)))
    return
    # predict on train data
    accu_train,d_Y_pred_train=predict(w,b,d_X_train,d_Y_train)
    # predict on test data
    accu_test,d_Y_pred_test=predict(w,b,d_X_test,d_Y_test)
    # get confusion matrix for train data
    conf_train=confusion_matrix(d_Y_train,d_Y_pred_train)
    f1_train=f1_score(d_Y_train,d_Y_pred_train,average=None)
    macro_f1_train=f1_score(d_Y_train,d_Y_pred_train,average='macro')
    # get confusion matrix for test data
    conf_test=confusion_matrix(d_Y_test,d_Y_pred_test)
    f1_test=f1_score(d_Y_test,d_Y_pred_test,average=None)
    macro_f1_test=f1_score(d_Y_test,d_Y_pred_test,average='macro')
    
    # save linear cvxopt results to "svm_linear_cvxopt.txt"
    with open("svm_linear_cvxopt.txt", "w") as f:
        f.write("Training time: "+str(end-start)+"\n")
        f.write("Number of support vectors: "+str(nSV)+"\n")
        f.write("Accuracy on train data: "+str(accu_train)+"\n")
        f.write("F1 score on train data: "+str(f1_train)+"\n")
        f.write("Macro F1 score on train data: "+str(macro_f1_train)+"\n")
        f.write("Confusion matrix on train data:\n")
        for i in range(len(conf_train)):
            for j in range(len(conf_train[0])):
                f.write(str(conf_train[i][j])+" ")
            f.write("\n")
        f.write("Accuracy on test data: "+str(accu_test)+"\n")
        f.write("F1 score on test data: "+str(f1_test)+"\n")
        f.write("Macro F1 score on test data: "+str(macro_f1_test)+"\n")
        f.write("Confusion matrix on test data:\n")
        for i in range(len(conf_test)):
            for j in range(len(conf_test[0])):
                f.write(str(conf_test[i][j])+" ")
            f.write("\n")
    # save confusion matrix plots
    draw_confusion(conf_train,"_cvxopt_gauss_train")
    draw_confusion(conf_test,"_cvxopt_gauss_test")

def libsvm_linear_gaussian (train_path,test_path):
    # read data
    X_train,Y_train=read_data(train_path)
    X_test,Y_test=read_data(test_path)
    # get subset of data relevant to digit d and return X and Y
    d_X_train,d_Y_train=get_d_X_Y(X_train,Y_train,digit1,digit2)
    d_X_test,d_Y_test=get_d_X_Y(X_test,Y_test,digit1,digit2)
    
    # model problem
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

###########################################################################

# functions for part 2
def multiclass_cvxopt(train_path,test_path,part="c"):
    # read data
    X_train,Y_train=read_data(train_path)
    X_test,Y_test=read_data(test_path)
    # default dictionary
    final_pred_train=[collections.defaultdict(int) for i in range(len(Y_train))]
    final_pred_test=[collections.defaultdict(int) for i in range(len(Y_test))]
    train_time=0
    for digit1 in range(10):
        for digit2 in range(digit1+1,10):
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
        # write results to "svm_cvxopt_multiclass.txt"
        with open("svm_cvxopt_multiclass.txt", "w") as f:
            f.write("Train time: "+str(train_time)+"\n")
            f.write("Multiclass CVXOPT Training accuracy: "+str(accu_train)+"\n")
            f.write("Multiclass CVXOPT Test accuracy: "+str(accu_test)+"\n")
            f.write("Multiclass CVXOPT Training confusion matrix: \n")
            f.write(str(conf_matrix_train)+"\n")
            f.write("Multiclass CVXOPT Test confusion matrix: \n")
            f.write(str(conf_matrix_test)+"\n")
        # plot confusion matrix
        draw_confusion(conf_matrix_train,"_cvxopt_multiclass_train")
        draw_confusion(conf_matrix_test,"_cvxopt_multiclass_test")



def multiclass_libsvm(train_path,test_path,part="c"):
    # read data
    X_train,Y_train=read_data(train_path)
    X_test,Y_test=read_data(test_path)
    # default dictionary
    final_pred_train=[collections.defaultdict(int) for i in range(len(Y_train))]
    final_pred_test=[collections.defaultdict(int) for i in range(len(Y_test))]
    train_time=0
    for digit1 in range(10):
        for digit2 in range(digit1+1,10):
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
    print("Multiclass LIBSVM Training accuracy: "+str(accu_train))
    accu_test=0
    for i in range(len(Y_test)):
        if Y_test[i]==final_pred_test_lbl[i]:
            accu_test+=1
        else:
            print("Missclassified",Y_test[i],"as",final_pred_test_lbl[i],"datapoint:",i)
    accu_test/=len(Y_test)
    print("Multiclass LIBSVM Test accuracy: "+str(accu_test))
    if part=="c":
        # confusion matrix
        conf_matrix_train=confusion_matrix(Y_train,final_pred_train_lbl)
        conf_matrix_test=confusion_matrix(Y_test,final_pred_test_lbl)
        # write results to "svm_libsvm_multiclass.txt"
        with open("svm_libsvm_multiclass.txt", "w") as f:
            f.write("Train time: "+str(train_time)+"\n")
            f.write("Multiclass LIBSVM Training accuracy: "+str(accu_train)+"\n")
            f.write("Multiclass LIBSVM Test accuracy: "+str(accu_test)+"\n")
            f.write("Multiclass LIBSVM Training confusion matrix: \n")
            f.write(str(conf_matrix_train)+"\n")
            f.write("Multiclass LIBSVM Test confusion matrix: \n")
            f.write(str(conf_matrix_test)+"\n")
        # plot confusion matrix
        draw_confusion(conf_matrix_train,"_libsvm_multiclass_train")
        draw_confusion(conf_matrix_test,"_libsvm_multiclass_test")

def cross_validation(train_path):
    # read data
    X_train,Y_train=read_data(train_path)
    C_set=[10,2,1,1e-5,1e-3]
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
    for C in C_set:
        # get accuracy
        accu_train=0
        accu_test=0
        train_time=0
        for i in range(K_fold):
            print("################### fold "+str(i+1)+" ###################")
            # get subset of data relevant to fold i
            X_train_fold=[]
            Y_train_fold=[]
            for j in range(K_fold):
                if j!=i:
                    X_train_fold+=X_train_folds[j]
                    Y_train_fold+=Y_train_folds[j]
            # model problem
            problem=svm_problem(Y_train_fold,X_train_fold)
            # set parameters
            param=svm_parameter('-s 0 -t 2 -g '+ str(gamma) +' -c '+str(C))
            # train model
            start=time.time()
            print("################ Starting training")
            model=svm_train(problem,param)
            train_time+=time.time()-start
            print("Training complete")
            # predict on train data
            pred_lbl,pred_acc_train,pred_val=svm_predict(Y_train_fold,X_train_fold,model)
            j=0
            for i in range(len(Y_train_fold)):
                if Y_train_fold[i]==pred_lbl[j]:
                    accu_train+=1
                j+=1
            # predict on test data
            pred_lbl,pred_acc_test,pred_val=svm_predict(Y_train_folds[i],X_train_folds[i],model)
            j=0
            for i in range(len(Y_train_folds[i])):
                if Y_train_folds[i][j]==pred_lbl[j]:
                    accu_test+=1
                j+=1
        accu_train/=len(Y_train)
        accu_test/=len(Y_train)
        print("Cross validation LIBSVM Training accuracy: "+str(accu_train))
        print("Cross validation LIBSVM Test accuracy: "+str(accu_test))
        # write results to "svm_libsvm_

# main function
if __name__ == '__main__':
    # get command line arguments
    train_path=sys.argv[1]
    test_path=sys.argv[2]
    class_num=int(sys.argv[3])
    part_num=sys.argv[4]
    # binary classification
    if class_num==0:
        if part_num=='a':
            linear_cvxopt(train_path,test_path)
            # tasks:
            # 1. obtain support vectors and save to file
            # 2. print bias
            # 3. print weights
            # 4. print validation accuracy
            # 5. print test accuracy
            # 6. print time required
    # exit()    
        elif part_num=='b':
            print("Starting gaussian cvxopt")
            gaussian_cvxopt(train_path,test_path)
            # tasks:
            # 1. obtain support vectors and save to file
            # 2. print validation accuracy
            # 3. print test accuracy
            # 4. print time required
        elif part_num=='c':
            libsvm_linear_gaussian(train_path,test_path)
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
            multiclass_libsvm(train_path,test_path,"c")
            # multiclass_cvxopt(train_path,test_path)
        elif part_num=='b':
            multiclass_libsvm(train_path,test_path,"b")
        elif part_num=='c':
            multiclass_libsvm(train_path,test_path,"c")
        elif part_num=='d':
            cross_validation(train_path)

        else:
            print('wrong part number')
            exit()
    else:
        print("class_num should be 0 or 1")
        exit()
