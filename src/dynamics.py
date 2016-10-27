import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import brute
from scipy.optimize import fmin
from scipy.special import beta as beta_func
from scipy.special import binom
from scipy.misc import comb
from scipy.stats import chi2
from functools import partial



# Define the discrete-time replicator dynamics
def discrete_time_replicator_dynamics(n_steps, X, Y, A, B, P):
    """Simulate the discrete-time replicator dynamics.

    Parameters
    ----------
    n_steps : int, the number of discrete time steps to simulate
    X : stochastic sender matrix
    Y : stochastic receiver matrix
    A : sender utility matrix
    B : receiver utility matrix
    P : prior probability over states matrix

    Returns
    ----------
    X_t : array-like, the state of the sender population at each year
    Y_t : array-like, the state of the receiver population at each year
    """
    # Get the number of states
    X_nrow = X.shape[0]
    # Get the number of messages
    X_ncol = X.shape[1]
    # Get the number of actions
    Y_nrow = Y.shape[0]
    Y_ncol = Y.shape[1]
    # Create empty arrays to hold flattened matrices for the population over time
    X_t = np.empty(shape=(n_steps, X_nrow*X_ncol), dtype=float)
    Y_t = np.empty(shape=(n_steps, X_nrow*X_ncol), dtype=float)
    # Set the initial state
    X_t[0,:] = X.ravel()
    Y_t[0,:] = Y.ravel()
    # Iterate forward over (n-1) steps
    for i in range(1,n_steps):
        # Get the previous state
        X_prev = X_t[i-1,:].reshape(X_nrow, X_ncol)
        Y_prev = Y_t[i-1,:].reshape(Y_nrow, Y_ncol)
        # Calculate the scaling factors
        E_X = A * Y_prev.T
        X_bar = (((A * Y_prev.T) * X_prev.T).diagonal()).T
        X_hat = E_X / X_bar
        # Calculate probability of states given messages
        C = np.divide(np.multiply(P.T, X_prev), (P * X_prev)[0])
        E_Y = (B.T * C).T
        Y_bar = ((E_Y*Y_prev.T).diagonal()).T
        Y_hat = np.divide(E_Y, Y_bar)
        # Calculate current states
        X_t[i,:] = np.multiply(X_prev, X_hat).ravel()
        Y_t[i,:] = np.multiply(Y_prev, Y_hat).ravel()
    return X_t, Y_t


# Define various components used to construct the model

def beta_binomial(alpha, beta, n=100):
    return np.matrix([comb(n-1,k) * beta_func(k+alpha, n-1-k+beta) / beta_func(alpha,beta) for k in range(n)])

def U_S(state, action, b):
    return 1 - (action - state - (1-state)*b)**2
def U_R(state, action):
    return 1 - (action - state)**2

def t(i, n):
    return i/float(n)
def a(i, n):
    return i/float(n)

def sender_matrix(b, number=100):
    return np.matrix([[U_S(t(i, number-1), a(j,number-1), b) for j in range(number)] for i in range(number)])
def receiver_matrix(number=100):
    return np.matrix([[U_R(t(i, number-1), a(j,number-1)) for j in range(number)] for i in range(number)])


def construct_initial_state(a_s, b_p, b=0):
    """Construct the initial state of the model.

    Parameters
    ----------
    a_s, b_p, b : parameters defined in document

    Returns
    ----------
    X0 : array-like, the initial state of the speaker population
    Y0 : array-like, the initial state of the hearer population
    prior : prior probability over states
    """
    # Define prior probability
    a_p = 1
    prior = beta_binomial(a_p, b_p)
    P = np.repeat(prior, 2, axis=0)
    # Define payoff matrices
    A = sender_matrix(b)
    B = receiver_matrix(    )
    # Define speaker population
    X0_m2 = beta_binomial(a_s, 1)
    X0_m1 = 1 - X0_m2
    X0 = np.vstack((X0_m1, X0_m2)).T
    # Calculate probability of state given m2
    p_ti_m2 = np.multiply(X0[:,1], prior.T)
    p_m2 = prior * X0[:,1]
    p_t_m2 = p_ti_m2 / p_m2
    # Calculate probability of state given m1
    p_ti_m1 = np.multiply(X0[:,0], prior.T)
    p_m1 = prior * X0[:,0]
    p_t_m1 = p_ti_m1 / p_m1
    # Calculate expected utility for receiver of action given m1
    E_ai_m1 = p_t_m1.T * B
    E_a_m1 = E_ai_m1 / E_ai_m1.sum()
    # Calculate expected utility for receiver of action given m2
    E_ai_m2 = p_t_m2.T * B
    E_a_m2 = E_ai_m2 / E_ai_m2.sum()
    # Define hearer population
    Y0 = np.vstack([E_a_m1, E_a_m2])

    return X0, Y0, A, B, prior


def simulate_dynamics(params, n_years=401, time_scale=1, number=100):
    """Simulate the discrete-time behavioral replicator dynamics for the game.

    Parameters
    ----------
    n_years : int, the number of discrete time steps to simulate
    time_scale : int (optional), the number of discrete time steps per year
    number : int, (optional), the number of discretized states and actions
    params : array-like, parameters that determine starting state of population

    Returns
    ----------
    X_sol : array-like, the state of the speaker population at each year
    Y_sol : array-like, the state of the hearer population at each year
    prior : prior probability over states
    """
    # Unpack the parameters
    a_s, b_p, b = params
    # Construct the initial state
    X0, Y0, A, B, prior = construct_initial_state(a_s, b_p, b)
    # Create prior probability matrix
    P = np.repeat(prior, 2, axis=0)
    # Iterate using dynamics to get values for the number of years
    X_sol, Y_sol = discrete_time_replicator_dynamics(n_years*time_scale, X0, Y0, A, B, P)
    X_sol = X_sol[0::time_scale,:]
    Y_sol = Y_sol[0::time_scale,:]
    return X_sol, Y_sol, prior


func_data = pd.read_csv('../data/functional-cycle-data.csv')

def loss_function(params, func=simulate_dynamics, time_scale=1, df=func_data):
    """Calculate the loss function.

    Parameters
    ----------
    params : array-like, parameters that determine starting state of population
    func : model begin fit to the data
    time_scale : int (optional), the number of discrete time steps per year
    number : int, (optional), the number of discretized states and actions
             default is one "generation" of the dynamics per year
    df : data to use, default is from func_data

    Returns
    ----------
    negLL : float, negative log likelihood to be minimized
    """
    # Simulate the dynamics from 1100 to 1500
    X_sol, Y_sol, prior = func(params, n_years=401, time_scale=time_scale)
    # Get p(m_2) over time
    m2_sol = np.asarray([prior.dot(line)[0,0] for line in X_sol[:,1::2]])
    # Append solution trajectory to data frame
    df['p'] = m2_sol.ravel()
    # Add binomial coefficient
    df['binom'] = binom(df.ones + df.zeros, df.ones)
    # Calculate log-likelihood for
    df['LL'] = np.log(df.binom) + (df.ones * np.log(df.p)) + (df.zeros * np.log(1 - df.p))
    # Only use years that have tokens
    df = df[df['has.tokens'] == 1]
    # Calculate log likelihood given m2_sol
    LL = np.sum(df['LL'])
    negLL = -1*LL
    # Minimizing negative log-likelihood is equivalent to maximizing log-likelihood
    return negLL


def simulate_simplified_dynamics(params, n_years=401, time_scale=1, number=100):
    """Simulate simplified dynamic model.

    Parameters
    ----------
    n_years : int, the number of discrete time steps to simulate
    time_scale : int (optional), the number of discrete time steps per year
    number : int, (optional), the number of discretized states and actions
    params : array-like, parameters that determine starting state of population

    Returns
    ----------
    X_sol : array-like, the state of the speaker population at each year
    Y_sol : array-like, the state of the hearer population at each year
    prior : prior probability over states
    """
    # Unpack the parameters
    a_s, b_p = params
    # Construct the initial state
    X0, Y0, A, B, prior = construct_initial_state(a_s, b_p)
    # Create prior probability matrix
    P = np.repeat(prior, 2, axis=0)
    # Iterate using dynamics to get values for the number of years
    X_sol, Y_sol = discrete_time_replicator_dynamics(n_years*time_scale, X0, Y0, A, B, P)
    X_sol = X_sol[0::time_scale,:]
    Y_sol = Y_sol[0::time_scale,:]
    return X_sol, Y_sol, prior


def main():
    rranges = (slice(1, 50, 10), slice(1, 50, 10), slice(0, 1, 0.25))
    model_results = brute(loss_function, rranges, finish=fmin)
    print model_results

    simplified_loss_function = partial(loss_function, func=simulate_simplified_dynamics)
    rranges = (slice(1, 20, 1), slice(1, 20, 1))
    simplified_results = brute(simplified_loss_function, rranges, finish=fmin)
    print simplified_results

    full_LL = -1 * loss_function(model_results)
    full_AIC = 2*3 - 2*full_LL

    simple_LL = -1*loss_function(simplified_results, func=simulate_simplified_dynamics)
    simple_AIC = 2*2 - 2*simple_LL


    delta_AIC =  simple_AIC - full_AIC
    print "Difference in AIC between models is %f" % delta_AIC


    D = 2*(full_LL - simple_LL)
    chi2_result = chi2.pdf(D,1)
    print "LRT test statistic D = %0.03f, with 1 d.o.f (p < %0.03f)" % (D, chi2_result)


if __name__ == "__main__":
    main()
