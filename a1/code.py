import matplotlib.pyplot as plt
import numpy

# gradient descent calculator
def gradient_descent(X, Y, theta, alpha, m, iterations):
    thetaT = theta.transpose()
    for i in range(iterations):
        h = numpy.dot(thetaT, x)
        loss = h - y
        cost = numpy.sum(loss ** 2) / (2 * m)
        print("Iteration %d | Cost: %f" % (i, cost))
        gradient = numpy.dot(xTrans, loss) / m
        theta = theta - alpha * gradient
    return theta

# Main
if __name__ == "__main__":
    # Load data
    X = numpy.loadtxt("data/q1/linearX.csv", delimiter=",")
    Y = numpy.loadtxt("data/q1/linearY.csv", delimiter=",")
    m = len(Y)

    # Plot data
    plt.scatter(x, y, marker="x", color="r")
    plt.xlabel("Population of City in 10,000s")
    plt.ylabel("Profit in $10,000s")

    # Calculate theta
    theta = numpy.array([0, 0])
    iterations = 1500
    alpha = 0.01
    theta = gradient_descent(x, y, theta, alpha, m, iterations)
    print("theta: ", theta)

    # Plot line
    plt.plot(x, theta[0] + theta[1] * x, color="b")

    plt.show()