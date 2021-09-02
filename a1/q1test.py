# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm



# %%

X=pd.read_csv("data/q1/linearX.csv",header=None)
Y=pd.read_csv("data/q1/linearY.csv",header=None)
m=len(X)
X = np.array(X)
Y = np.array(Y)
mean=np.mean(X)
var=np.var(X)**0.5
X=(X-mean)/var
X1=X
plt.scatter(X,Y)


# %%

X=np.c_[np.ones(m),X]
theta=np.zeros((X.shape[1],1))
alpha=1
converged=False
cost_last=-1
epsilon=1e-6
ans=0


# %%
plotTheta=[]
plotCost=[]
while not converged:
    ans+=1
    H = np.dot(X,theta)
    loss=np.dot((H-Y).T,(H-Y))
    grad=np.dot(X.T,(H-Y))/(2*m)
    theta=theta-alpha*grad
    cost=loss/(2*m)
    if abs(cost_last-cost)<epsilon:
        converged=True
    cost_last=cost
    plotTheta.append(theta)
    plotCost.append(cost[0,0])
print(theta)
plt.scatter(X1,Y)
plt.plot(X1,H)
plt.show()


# %%
def cost(X,Y,theta):
    m=len(X)
    H = np.dot(X,theta)
    loss=np.dot((H-Y).T,(H-Y))
    cost=loss/(2*m)
    return cost[0,0]
x1_size=100
x2_size=100
theta1sample=np.linspace(0,2,x1_size)
theta2sample=np.linspace(-1,1,x2_size)
X1,X2=np.meshgrid(theta1sample,theta2sample)
J = np.asmatrix(np.zeros((x1_size,x2_size),dtype=float))
for i in range(x1_size):
    for j in range(x2_size):
        xx=cost(X,Y,[[X1[i][j]],[X2[i][j]]])
        J[i,j] = xx


# %%

ax=plt.axes(projection='3d')
cp = ax.plot_surface(X1,X2,np.array(J),cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax.set_xlabel('theta1')
ax.set_ylabel('theta2')
ax.set_zlabel('cost')
theta1=np.array(plotTheta)[:,0]
theta2=np.array(plotTheta)[:,1]
plt.ion()
for i in range(len(plotCost)):
    ax.scatter(theta1[i],theta2[i],plotCost[i],color='red')
    plt.pause(0.1)
plt.ioff()
plt.show()

# %%
cp=plt.contour(X1,X2,np.array(J),50)
plt.ion()
for i in range(len(plotCost)):
    plt.scatter(theta1[i],theta2[i],color='red')
    plt.pause(0.1)
plt.ioff()
plt.show()


# %%
print(plotCost)


# %%



