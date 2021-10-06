import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
np.random.seed(0)

# Sample 1 million data points from a normal distribution
def sample_data(size):
    x1 = np.random.normal(3,2,size=size)
    x2 = np.random.normal(-1,2,size=size)
    theta=np.array([[3,1,2]]).T
    X=np.c_[np.ones(size),x1,x2]
    epsilon=np.random.normal(0,2**0.5,size=size)
    epsilon=np.c_[epsilon]
    Y=np.dot(X,theta)+epsilon
    
    # Y=np.dot(X,theta)
    return X,Y

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def read_data(path="../data/q2/logisticX.csv"):
    X = np.array(pd.read_csv(Xpath,header=None).values)
    Y = np.array(pd.read_csv(Ypath,header=None).values)
    return X,Y
def augment_intecept(X):
    m=len(X)
    return np.c_[np.ones(m),X]
# Implement stochastic gradient descent
def sgd(theta,X,Y,alpha,max_iter,r,epsilon,checkpoint):
    m=len(Y)
    X,Y=unison_shuffled_copies(X,Y)
    batches=m//r
    converged=False
    iter=0
    cost_last=-1
    costvector=[-1]
    thetagraph=[]
    temp=0
    while not converged:
        
        for b in range(batches):
            print("Current iteration: ",iter,cost_last)
            iter+=1
            if b%checkpoint==0:
                costvector+=[temp/checkpoint]
                temp=0
                if abs(costvector[-1]-costvector[-2])<epsilon:
                    print(costvector)
                    converged=True
                    break
            X_b=X[b*r:(b+1)*r]
            # print(X_b)
            Y_b=Y[b*r:(b+1)*r]
            H = np.dot(X_b,theta)
            loss=np.dot((H-Y_b).T,(H-Y_b))
            # print(X_b.T.shape,H.shape,Y_b.shape,(H-Y_b).shape)
            grad=np.dot(X_b.T,(H-Y_b))/(2*r)
            theta=theta-alpha*grad
            if b%10==0:
                thetagraph+=[theta]
            if(b!=batches-1):
                theta=theta-(alpha/m)*np.dot(X[(b-1)*r:b*r].T,(np.dot(X[(b-1)*r:b*r],theta)-Y[(b-1)*r:b*r]))
            else:
                theta=theta-(alpha/m)*np.dot(X[(b-1)*r:].T,(np.dot(X[(b-1)*r:],theta)-Y[(b-1)*r:]))
            cost=loss/(2*r)
            cost=cost[0,0]
            temp+=cost
            cost_last=cost
            if iter>max_iter:
                converged=True
                break
    plt.plot(costvector[2:])
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Cost vs Iterations")
    plt.legend(["GD","SGD"])
    plt.savefig("assets/xxx.svg")
    return theta,iter,thetagraph

# Main
if __name__ == "__main__":
    X,Y=sample_data(1000000)
    theta,iter,thetagraph=sgd([[0],[0],[0]],X,Y,alpha=0.001,max_iter=5000,r=10000,epsilon=1e-9,checkpoint=10)
    theta,iter,thetagraph=sgd([[0],[0],[0]],X,Y,alpha=0.001,max_iter=25000,r=1,epsilon=1e-9,checkpoint=50)
    print("Theta: ",theta)
    print("Total Iterations: ",iter)
    # 3d plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter([c[0] for c in thetagraph],[c[1] for c in thetagraph],[c[2] for c in thetagraph],c='r',marker='.')
    ax.set_xlabel(r'$\theta_0$')
    ax.set_ylabel(r'$\theta_1$')
    ax.set_zlabel(r'$\theta_2$')
    ax.set_title(r'Progress of $\theta$ in GD')
    plt.savefig("assets/3dprogress.svg")
    plt.show()
    H=np.dot(X,theta)
    loss=np.dot((H-Y).T,(H-Y))/(2*1000000)
    print("Loss: ",loss[0,0])

    test_data = np.genfromtxt('../data/q2/q2test.csv', delimiter=',')
    test_data = test_data[1:,:]
    X_test = test_data[:, :-1]
    Y_test = test_data[:, -1].reshape(-1,1)
    X_test=augment_intecept(X_test)
    print(X_test.shape,Y_test.shape)
    H_test=np.dot(X_test,theta)
    loss_test=np.dot((H_test-Y_test).T,(H_test-Y_test))/(2*len(Y_test))
    print("Loss on test data: ",loss_test[0,0])

    H_bias=np.dot(X_test,[[3],[1],[2]])
    loss_bias=np.dot((H_bias-Y_test).T,(H_bias-Y_test))/(2*len(Y_test))
    print("Loss on test data with bias: ",loss_bias[0,0])
