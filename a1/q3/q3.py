
from operator import le
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def read_data(Xpath="../data/q3/logisticX.csv",Ypath="../data/q3/logisticY.csv"):
    X = np.array(pd.read_csv(Xpath,header=None).values)
    Y = np.array(pd.read_csv(Ypath,header=None).values)
    X = normalize(X)
    return X,Y

def normalize(X):
    mean=np.mean(X,axis=0)
    X=(X-mean)/np.std(X,axis=0)
    return X

def augment_intecept(X):
    m=len(X)
    return np.c_[np.ones(m),X]

def get_cost(X,Y,theta):
    m=len(X)
    H = np.dot(X,theta)
    loss=np.dot((H-Y).T,(H-Y))
    cost=loss/(2*m)
    return cost[0,0]

def h_theta(X,theta):
    Z = np.dot(X,theta)
    H = 1/(1+np.exp(-Z))
    return H
def grad_l_theta(X,Y,theta):
    H = h_theta(X,theta)
    grad=np.dot(X.T,(Y-H))
    return grad

# logistic regression
def logistic_regression(X,Y,alpha=0.1,threshold=0.00000001,max_iter=10):
    theta=np.zeros((X.shape[1],1))
    converged=False
    costvector=[-1]
    iter=0
    while not converged:
        theta = theta + alpha*grad_l_theta(X,Y,theta)
        costvector.append(get_cost(X,Y,theta))
        if abs(costvector[-1]-costvector[-2])<threshold or iter>max_iter:
            converged=True
    return theta,costvector[1:]

# hessian calculation
def hessian(x,Y,theta):
    h=h_theta(x,theta)
    D=np.diag((h*(1-h)).flatten())
    l_d=-np.dot(x.T,(Y-h))/(2*Y.shape[0])
    l_d_d=np.dot(np.dot(x.T,D),x)/(2*Y.shape[0])
    return l_d,l_d_d

# netwons method
def netwons_method(X,Y,maxiter=10,threshold=0.001):
    theta=np.zeros((X.shape[1],1))
    l_d,l_d_d=hessian(X,Y,theta)
    converged=False
    costvector=[-1]
    iter=0
    while not converged:
        iter+=1
        theta = theta - np.dot(np.linalg.pinv(l_d_d),l_d)
        cost=get_cost(X,Y,theta)
        costvector.append(cost)
        if(cost<threshold or abs(costvector[-1]-costvector[-2])<threshold or iter>maxiter):
            converged=True
    return theta,costvector[1:]

def get_accuracy(X,Y,theta):
    Y_pred=h_theta(X,theta)
    Y_pred=np.where(Y_pred>0.5,1,0)
    return np.mean(Y_pred==Y)
X,Y=read_data()
X=augment_intecept(X)

plt.title("Logistic Regression")
plt.xlabel(r'$X_1$')
plt.ylabel(r'$X_2$')

theta,costvector=netwons_method(X,Y,100,0.00001)
x=np.linspace(-3,3,2)
y=-(theta[0,0]+theta[1,0]*x)/theta[2,0]
plt.plot(x,y)
zer=[]
one=[]
for i in range(len(Y)):
    if Y[i]==0:
        zer+=[X[i]]
    else:
        one+=[X[i]]
plt.scatter([c[1] for c in zer],[c[2] for c in zer],c='red',label='0',marker='x')
plt.scatter([c[1] for c in one],[c[2] for c in one],c='blue',label='1',marker='o')

print("Gradient ascent Method:")
print("Theta:",theta)
print("accuracy:",get_accuracy(X,Y,theta))
print("c:",theta[0,0]/theta[2,0],"m:",theta[1,0]/theta[2,0])

print("Newton's Method:")
# theta,costvector=netwons_method(X,Y,100,0.00001)
theta,costvector=logistic_regression(X,Y)
print("Theta:",theta)
# plot line corresponding to theta
x=np.linspace(-3,3,2)
y=-(theta[0,0]+theta[1,0]*x)/theta[2,0]
plt.plot(x,y)
print("accuracy:",get_accuracy(X,Y,theta))
print("c:",theta[0,0]/theta[2,0],"m:",theta[1,0]/theta[2,0])
plt.legend(['Newton\'s Method','Gradient ascent','Zero','One'])
plt.savefig("assets/logistic-reg.png")
plt.show()

