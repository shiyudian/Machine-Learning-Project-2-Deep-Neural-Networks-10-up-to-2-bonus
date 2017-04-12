from numpy import *
from numpy.linalg import norm
from matplotlib.pyplot import *
import scipy.stats


def plot_line(theta, x_min, x_max, color, label):
    x_grid_raw = arange(x_min, x_max, 0.01)
    x_grid = vstack((    ones_like(x_grid_raw),
                         x_grid_raw,
                    ))
    y_grid = dot(theta, x_grid)
    plot(x_grid[1,:], y_grid, color, label=label)

def gradf(f,t,error=1e-11):
  t = asarray(t)
  return asarray((f(t+error)-f(t-error))/(2*error))

def grad_descent2(f,init_t, alpha=1e-6):
  init_t = asarray(init_t)
  if alpha<1e-4:
    alpha = 1e-4

  EPS = 1e-8
  prev_t = init_t.copy()

  prev_t = (init_t-10*EPS).copy()
  t = init_t.copy()

  max_iter = 5000
  iter_n = 0

  residue0 = norm(t - prev_t)
  residue1 = norm(t - prev_t)

  factor0 = 1.22
  b_fac = 1.3
  factor1 = 0.5

  for i in range(max_iter):
  #while  residue1 > EPS and iter_n < max_iter:
    residue0 = residue1

    residue0 = residue1
    prev_t = t.copy()
    #t -= gradf(f,t)*alpha
    t -= gradf(f,t)*alpha
    iter_n += 1

    residue1 = norm(t - prev_t)
    #if f(t)-f(prev_t) < ala_r*dot(t-pre_t,t-pre_t)/alpha:
    if residue1 < residue0*factor0:
      #print "Boost"
      alpha = alpha*b_fac
      if alpha> 0.5:
        alpha = 0.5

    if f(t) > f(prev_t):
      #print "Slow"
      if norm(t)*alpha > EPS*100:
        alpha = alpha*factor1
        t = prev_t.copy()

    if residue1 < EPS:
      break

  if iter_n >= max_iter-1:
    print [alpha,t,residue1]
    print "Overloaded"
  return [alpha,t]

#print grad_descent2(cos,3.0,3e-5)

def gen_lin_data_1d(theta, N, sigma):

    #####################################################
    # Actual data
    x_raw = 100*(random.random((N))-.5)
    
    x = vstack((    ones_like(x_raw),
                    x_raw,
                    ))
                
    y = dot(theta, x) + scipy.stats.norm.rvs(scale= sigma,size=N)
    #saveJ(x,"tem_x")
    #saveJ(x,"tem_y")

    plot(x[1,:], y, "ro", label = "Training set")
    #####################################################
    # Actual generating process
    #
    plot_line(theta, -70, 70, "b", "Actual generating process")
    
    
    #######################################################
    # Least squares solution
    #
    theta_hat = dot(linalg.inv(dot(x, x.T)), dot(x, y.T))
    plot_line(theta_hat, -70, 70, "g", "Maximum Likelihood Solution")

    def C_fun(a_0):
      tem_l = maximum(0.0,1/(1+abs(dot(asarray(a_0), x))))
      tem_y = maximum(0.0,1/(1+abs(y)))
      return -sum(dot(tem_y,log(tem_l))+dot((1-tem_y),log(1-tem_l)))


    a,theta = grad_descent2(C_fun,(-3.1,1.1))
    plot_line(theta, -70, 70, "r", "Multinomial Logistic Regression")


    legend(loc = 1)
    xlim([-70, 70])
    ylim([-100, 100])

theta = array([-3, 1.5])
N = 10
sigma = 120.2

gen_lin_data_1d(theta, N, sigma)
show()

