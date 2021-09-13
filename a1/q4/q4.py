import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm


def normalize(X):
    mean=np.mean(X,axis=0)
    X=(X-mean)/np.var(X,axis=0)**0.5
    return X

def read_data(Xpath="../data/q4/q4x.dat",Ypath="../data/q4/q4y.dat"):
    X = np.array(pd.read_csv(Xpath,header=None,delim_whitespace=True).values)
    Y = np.array(pd.read_csv(Ypath,header=None,delim_whitespace=True).values)
    X = normalize(X)
    return X,Y

# Implement Gaussian Discriminant Analysis
def gda(X,Y):
    m=X.shape[0]
    psi=np.mean(np.where(Y=="Alaska",0,1))
    class0=(Y=="Alaska").flatten()
    class1=(Y=="Canada").flatten()
    mu_0=np.mean(X[class0,:],axis=0)
    mu_1=np.mean(X[class1,:],axis=0)
    temp0=X[class0,:]-mu_0
    temp1=X[class1,:]-mu_1
    sigma_0=np.dot(temp0.T,temp0)/len(temp0)
    sigma_1=np.dot(temp1.T,temp1)/len(temp1)
    sigma=(len(temp0)*sigma_0+len(temp1)*sigma_1)/m
    return psi,mu_0,mu_1,sigma_0,sigma_1,sigma

def plot_linear(mu_0,mu_1,sigma):
    const_term = np.dot(np.dot(mu_0.transpose(),np.linalg.pinv(sigma)),mu_0) - np.dot(np.dot(mu_1.transpose(),np.linalg.pinv(sigma)),mu_1)
    x_coeff = 2*np.dot((mu_1.transpose()-mu_0.transpose()),np.linalg.pinv(sigma))
    x1 = np.linspace(-2,2,2)
    print(const_term,x_coeff)
    x2 = -1*(const_term+x_coeff[0]*x1)/x_coeff[1]
    plt.plot(x1,x2,label='Linear Hypothesis')

def plot_quadratic(mu_0, mu_1, sigma_0,sigma_1):
    C1=np.log(((1-psi)*np.linalg.det(sigma_1)**0.5)/(psi*np.linalg.det(sigma_0)**0.5))
    C2=0.5*(np.dot(np.dot(mu_1.transpose(),np.linalg.pinv(sigma_1)),mu_1)-np.dot(np.dot(mu_0.transpose(),np.linalg.pinv(sigma_0)),mu_0))
    A=0.5*(np.linalg.pinv(sigma_1)-np.linalg.pinv(sigma_0))
    B=-1*(np.dot(mu_1.transpose(),np.linalg.pinv(sigma_1))-np.dot(mu_0.transpose(),np.linalg.pinv(sigma_0)))
    a=A[0][0]+A[0][1]
    b=A[1][0]+A[1][1]
    c=A[0][0]+A[0][1]+A[1][0]+A[1][1]
    d=B[0]
    e=B[1]
    f=C1+C2
    print("a=",a,"b=",b,"c=",c,"d=",d,"e=",e,"f=",f)
    x1=np.linspace(-3,3,200)
    x2=np.linspace(-3,3,200)
    x1,x2=np.meshgrid(x1,x2)
    return plt.contour(x1,x2,(a*x1**2+b*x2**2+c*x1*x2+d*x1+e*x2+f),[0],colors='m')

# main function
if __name__ == "__main__":
    X,Y=read_data()
    Y_pred=np.where(Y=="Canada",1,0)
    plt.legend(["Canada","Alaksa"])
    # plot points
    Canada=[]
    Alaska=[]
    for i in range(Y_pred.shape[0]):
        if Y_pred[i]==1:
            Canada+=[[X[i,0],X[i,1]]]
        else:
            Alaska+=[[X[i,0],X[i,1]]]

    plt.scatter([c[0] for c in Canada],[c[1] for c in Canada],label="Canada",c="red",marker='x')
    plt.scatter([c[0] for c in Alaska],[c[1] for c in Alaska],label="Alaska",c='blue',marker='o')

    psi,mu_0,mu_1,sigma_0,sigma_1,sigma=gda(X,Y)
    # print parameters
    print("psi:",psi)
    print("mu_0:",mu_0)
    print("mu_1:",mu_1)
    print("sigma_0:",sigma_0)
    print("sigma_1:",sigma_1)
    print("sigma:",sigma)
    
    # plot linear hypothesis
    plot_linear(mu_0,mu_1,sigma)
    # plot quadratic hypothesis
    t=plot_quadratic(mu_0, mu_1, sigma_0,sigma_1)
    plt.xlabel("Fresh Water")
    plt.ylabel("Marine Water")
    plt.title("Gaussian Discriminant Analysis")
    plt.plot([],[],color='m',label=r'$\Sigma_0 \neq \Sigma_1$')[0]
    plt.legend()
    plt.savefig("assets/gda.png")
    plt.show()
