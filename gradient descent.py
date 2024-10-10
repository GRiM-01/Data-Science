import numpy as np 
import matplotlib.pyplot as plt  
from typing import List 

Vector = List[float] 
figure, position = plt.subplots(2, 3)

def get_loss(w: float, xvals: np.ndarray, yvals: np.ndarray) -> float:    
    return np.sum((w * xvals + w * (xvals**2) - yvals)**2)

def get_gradient(w: float, xvals: np.ndarray, yvals: np.ndarray) -> float: 
    grad = np.sum(2 * (xvals + xvals**2) * (w * xvals + w * xvals**2 - yvals))     
    return grad 

def perform_gradient_descent(init_w: float, eta: float, num_iters: int) -> Vector: 
    w_vals = [init_w]      
    for _ in range(num_iters): 
        grad = get_gradient(w_vals[-1], xvals, yvals)         
        w_vals.append(w_vals[-1] - eta * grad)     
    return w_vals 


numvals = 50
tru_w = 0.5 
 
xvals = np.sort(5 * (np.random.random(numvals) - 0.5)) 
yvals = tru_w * (xvals + xvals**2) 
  


position[0, 0].scatter(xvals, yvals)
position[0, 0].set_title('Data') 

wvals = np.arange(0, 1.5, 0.1) 
lossvals = [get_loss(w, xvals, yvals) for w in wvals] 
position[0, 1].plot(wvals, lossvals) 
position[0, 1].set_title('Loss Function')

gradvals = [get_gradient(w, xvals, yvals) for w in wvals] 
position[0, 2].plot(wvals, gradvals) 
position[0, 2].axhline(0, color='k') 
position[0, 2].set_title('Gradient') 

w_vals = perform_gradient_descent(0, 0.0001, 50) 
position[1, 0].plot(w_vals)
position[1, 0].set_title(f'Gradient Descent Progress\nFinal w: {round(w_vals[-1], 2)}') 

xrange = np.arange(xvals.min(), xvals.max(), 0.01) 
pred_yvals_loss1 = w_vals[-1] * (xrange + xrange**2) 
position[1, 1].scatter(xvals, yvals) 
position[1, 1].plot(xrange, pred_yvals_loss1) 
position[1, 1].set_title('Fitted Model')

# Compare with the true model
true_yvals = tru_w * (xrange + xrange**2)
position[1, 1].plot(xrange, true_yvals, 'r--', label='True Model')
position[1, 1].legend()

plt.show()
