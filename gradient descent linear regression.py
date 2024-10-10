import numpy as np
import matplotlib.pyplot as plt


def hypothesis(theta_0, theta_1, x):
    return theta_0 + theta_1 * x

def cost(theta_0, theta_1, x, y):
    m = len(y)
    return (1/(2*m)) * np.sum((hypothesis(theta_0, theta_1, x) - y) ** 2)

def gradient_descent(x, y, theta_0, theta_1, alpha, num_iters):
    m = len(y)
    cost_history = []
    theta_history = []

    for _ in range(num_iters):
        cost_history.append(cost(theta_0, theta_1, x, y))
        theta_history.append((theta_0, theta_1))

        temp0 = theta_0 - alpha * (1/m) * np.sum(hypothesis(theta_0, theta_1, x) - y)
        temp1 = theta_1 - alpha * (1/m) * np.sum((hypothesis(theta_0, theta_1, x) - y) * x)
        theta_0, theta_1 = temp0, temp1

    return theta_0, theta_1, cost_history, theta_history


figure, position = plt.subplots(2, 2)

# Data
Hours = [2.5, 5.1, 3.2, 8.5, 3.5, 1.5, 9.2, 5.5, 8.3, 2.7, 7.7, 5.9, 4.5, 3.3, 1.1, 8.9, 2.5, 1.9, 6.1, 7.4, 2.7, 4.8, 3.8, 6.9, 7.6]
Scores = [21, 47, 27, 75, 30, 20, 88, 60, 81, 25, 85, 62, 41, 42, 17, 95, 30, 24, 67, 69, 30, 54, 35, 76, 84]
position[0,0].scatter(Hours, Scores)
position[0,0].set_title("Data")

Hours = np.array(Hours)
Scores = np.array(Scores)

theta_0 = 0
theta_1 = 0
alpha = 0.01
num_iters = 1000

theta_0, theta_1, cost_history, theta_history = gradient_descent(Hours, Scores, theta_0, theta_1, alpha, num_iters)

predicted_scores = hypothesis(theta_0, theta_1, Hours)

# Cost Function
position[0,1].plot(range(num_iters), cost_history, color='blue')
position[0,1].set_title('Cost Function (J) vs. Iterations')

# Gradient Descent
theta_0_vals, theta_1_vals = zip(*theta_history)
position[1,0].plot(range(num_iters), theta_0_vals, label=r'$\theta_0$', color='green')
position[1,0].plot(range(num_iters), theta_1_vals, label=r'$\theta_1$', color='orange')
position[1,0].set_title('Gradient Descent Progress')
                        
# Fitted Model
position[1,1].scatter(Hours, Scores, label='Fitted Model')
position[1,1].plot(Hours, predicted_scores, color='red', label=f'Fitted Line: h(x) = {round(theta_0, 2)} + {round(theta_1, 2)}*x')
position[1,1].set_title('Linear Regression Fit')

plt.show()

# Output final parameters
print(f"Final theta_0 (intercept): {theta_0}")
print(f"Final theta_1 (slope): {theta_1}")
