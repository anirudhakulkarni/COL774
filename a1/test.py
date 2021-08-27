import sys
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import linalg
from decimal import Decimal

# reading data from csv files
def read_params(x_path,y_path):
  x_data = pd.read_csv(x_path,header=None)
  y_data = pd.read_csv(y_path,header=None)
  x = normalize(np.asmatrix(np.array(x_data)))
  y = np.asmatrix(np.array(y_data))
  return (x,y,np.array(x_data),np.array(y_data))

# normalization function
def normalize(mat):
  mean = np.mean(mat,axis=0)
  var = np.var(mat,axis=0)
  mat-=mean
  np.divide(mat,var)

  # including the initial column of x0 feature
  z = 1 + np.zeros((mat.shape[0],1),dtype=float)
  z = np.hstack((z,mat))
  return z

# cost computation from current weights and outputs
def compute_cost(x,y,theta):
  diff = y-np.dot(x,theta)
  rv = np.dot(diff.transpose(),diff)
  return rv[0,0]/(2*len(x))

# gradient computation
def gradient(x,y,theta):
  grad = np.dot(np.dot(x.transpose(),x),theta) - np.dot(x.transpose(),y)
  return grad/len(x)

# main algorithm/
def algo(x,y,theta,alpha):
  num_iter = 0
  epsilon = 0.000000001
  old_cost = 1000
  new_cost = 0
  
  # contains the cost after each iteration
  cost_matrix = [[0,0,0]]
  
  # convergence criteria: Change in cost<epsilon
  while True:
    print(new_cost)
    grad = gradient(x,y,theta)
    theta-=alpha*grad
    new_cost = compute_cost(x,y,theta)
    if(old_cost-new_cost<epsilon):
      break
    old_cost = new_cost
    cost_matrix = np.append(cost_matrix,[[new_cost,theta[0,0],theta[1,0]]],axis=0)
    num_iter+=1
  
  # On console the number of iterations taken and the cost
  print("Number of iterations taken to converge= " + str(num_iter))
  print("Final cost = " + str(new_cost))

  # returning the cost martix along with the theta vector
  return np.asmatrix(cost_matrix)[1:,:]

# plotting the graph
def curve_plot(x,theta,x_data,y_data):
  plt.figure()
  y_out = np.array(np.dot(x,theta))
  plt.title('Linear regression')
  plt.scatter(x_data,y_data,c='r',label='Given Data')
  plt.xlabel(' x ')
  plt.ylabel(' y ')
  new_x,new_y = zip(*sorted(zip(x_data,y_out)))
  plt.plot(new_x,new_y,'-',linewidth=2,c='g',label='Hypothesis')
  plt.legend()
  plt.savefig('linear_reg.png',dpi=200)
  plt.show()

# 2D plotting the contours. Animated the process of convergence
def contour_plot(x_data,y_data,cost_data,time_gap):
  plt.figure()
  num_points_x1 = 100
  num_points_x2 = 100
  x1 = np.linspace(-1,1,num_points_x1)
  x2 = np.linspace(-1,1,num_points_x2)
  X1,X2 = np.meshgrid(x1,x2)
  Y = np.asmatrix(np.zeros((num_points_x1,num_points_x2),dtype=float))
  for i in range(num_points_x1):
    for j in range(num_points_x2):
      Y[i,j] = compute_cost(x_data,y_data,[[X1[i][j]],[X2[i][j]]])
  cp = plt.contour(X1,X2,np.array(Y))
  
  plt.xlabel('theta_0')
  plt.ylabel('theta_1')
  plt.ion()
  for i in range(len(cost_data)):
    plt.scatter(cost_data[i,1],cost_data[i,2],color='r') 
    plt.pause(time_gap)
  plt.ioff() 
  plt.show()

# 3D animation for simulating process of convergence
def mesh_plot(x_data,y_data,cost_data,time_gap):
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  num_points_x1 = 100
  num_points_x2 = 100
  x1 = np.linspace(-1,2,num_points_x1)
  x2 = np.linspace(-1,2,num_points_x2)
  X1,X2 = np.meshgrid(x1,x2)
  Y = np.asmatrix(np.zeros((num_points_x1,num_points_x2),dtype=float))
  for i in range(num_points_x1):
    for j in range(num_points_x2):
      Y[i,j] = compute_cost(x_data,y_data,[[X1[i][j]],[X2[i][j]]])
  cp = ax.plot_surface(X1,X2,np.array(Y),cmap=cm.coolwarm,linewidth=0, antialiased=False)
  
  plt.xlabel('theta_0')
  plt.ylabel('theta_1')
  plt.ion()
  for i in range(len(cost_data)):
    ax.scatter(cost_data[i,1],cost_data[i,2],cost_data[i,0],color='r') 
    plt.pause(time_gap)
  plt.ioff() 
  plt.show()


# MAIN FUNCTION
def main():
  # Taking parameters from command line
  x_path = sys.argv[1]
  y_path = sys.argv[2]
  alpha = float(sys.argv[3])
  time_gap = float(sys.argv[4])
  
  # Reading the dataset
  (x,y,x_data,y_data) = read_params(x_path,y_path)
  
  # Computing the theta vector and the cost vector
  theta = np.asmatrix(np.zeros((x.shape[1],1),dtype=float,order='F'))
  cost_matrix = algo(x,y,theta,alpha)
  
  # Plotting
  curve_plot(x,theta,x_data,y_data)
  contour_plot(x,y,cost_matrix,time_gap)
  mesh_plot(x,y,cost_matrix,time_gap)
  
if __name__ == "__main__":
	main()