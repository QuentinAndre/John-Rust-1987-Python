# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 18:16:50 2015

@author: Quentin
"""
import numpy as np
import scipy.optimize as opt

class DynamicLogit(object):
    def __init__(self, data, Y, X, p, MF, npars):
        """
        A statistics workbench used to evaluate the cost parameters underlying 
        a bus replacement pattern by a forward-looking agent.
        
        Takes:
            * Data: a Pandas dataframe, which contains:
                -Y: the name of the column containing the dummy exogenous 
                    variable (here, the choice)
                -X: the name of the column containing the endogenous variable 
                    (here, the state of the bus)

            * p: The state-transition vector of endogenous variable.
                    For instance, p = [0, 0.6, 0.4] means that the bus will 
                    transition to the next mileage state with probability 0.6, 
                    and to the second next mileage state with probability 0.4.

            * MF: A function passed as an argument, which is the functional 
                  form for the maintenance cost. This function must accept as
                  a first argument a state s, and as a second argument a vector
                  of parameters.
                  
            * npars: The number of parameters to evalutate (i.e. the number of 
                     parameters of the maintenance cost function, plus 1 for
                     the replacement cost)
        """        

        self.endog = data.loc[:, Y].values
        self.exog = data.loc[:, X].values
        
        self.N = self.endog.shape[0]
        self.S = int(self.exog.max()*2) # Assumes that the true maximum number 
                                         # states is twice the maximum observed
                                         # state.
        
        
        # Check that p is a correct vector of probabilities (i.e. sums to 1)        
        if p.sum() == 1:
            self.p = p
        else:
            raise ValueError(("The probability of state transitions should add" 
                              " up to 1!"))
        self.MF = MF
        self.npars = npars
        
        # Check that the stated number of parameters correspond to the
        # specifications of the maintenance cost function.       
        try:
            MF(1, [0]*npars)
        except ValueError:
            raise ValueError(("The number of parameters specified does not "
                              "match the specification of the maintenance cost"
                              " function!"))
        
        S = self.S        
        # To speed up computations and avoid loops when computing the log 
        # likelihood, we create a few useful matrices here:
        
        # A (SxN) matrix indicating the state of each observation is created 
        self.state_mat = np.array([[self.exog[i]==s for i in range(self.N)] 
                                                    for s in range(self.S)])
        
        # A (SxS) matrix indicating the probability of a bus transitioning
        # from a state s to a state s' (used to compute maintenance utility)
        
        self.trans_mat = np.zeros((S, S))
        for i in range(S):
            for j, _p in enumerate(self.p):
                if i + j < S-1:
                    self.trans_mat[i+j][i] = _p
                elif i + j == S-1:
                    self.trans_mat[S-1][i] = p[j:].sum()
                else:
                    pass

        # A second (SxS) matrix which regenerates the bus' state to 0 with
        # certainty (used to compute the replacement utility)
        self.regen_mat = np.vstack((np.ones((1, S)),np.zeros((S-1, S))))
        
        # A (2xN) matrix indicating with a dummy the decision taken for each
        # time/bus 
        self.dec_mat = np.vstack(((1-self.endog), self.endog))
    
        # A matrix d
    def myopic_costs(self, params):
        S = self.S
        """
        This function computes the myopic expected cost associated with each decision for each state.
        
        Takes:
            * A vector params, to be supplied to the maintenance cost function 
              MF. The first element of the vector is the replacement cost rc.

        Returns:
            * A (Sx2) array containing the maintenance and replacement costs 
              for the S possible states of the bus
        """
        rc = params[0]
        thetas = params[1:]
        maint_cost = [self.MF(s, thetas) for s in range(0, S)]
        repl_cost = [rc for state in range(0, S)]
        return np.vstack((maint_cost, repl_cost)).T
    
    def choice_prob(self, cost_array):
        """
        Returns the probability of each choice for each observed state, 
        conditional on an array of state/decision costs.
        """
        cost = cost_array - cost_array.min(1).reshape(self.S, -1)
        util = np.exp(-cost)
        pchoice = util/(np.sum(util, 1).reshape(self.S, -1))
        return pchoice
    
    def fl_costs(self, params, beta=0.75, threshold=1e-6, suppr_output=False):
        """
        Compute the non-myopic expected value of the agent for each possible 
        decision and each possible state of the bus, conditional on a vector of 
        parameters and on the maintenance cost function specified at the 
        initialization of the DynamicUtility model.

        Iterates until the difference in the previously obtained expected value 
        and the new expected value is smaller than a constant.
        
        Takes:
            * A vector params for the cost function
            * A discount factor beta (optional)
            * A convergence threshold (optional)
            * A boolean argument to suppress the output (optional)

        Returns:
            * An (Sx2) array of forward-looking costs associated with each
              sate and each decision.
        """
        achieved = True
        # Initialization of CM
        k = 0
        EV = np.zeros((self.S, 2))
        EV_myopic = EV_new = self.myopic_costs(params)
        
        # CM Loop
        while abs(EV_new-EV).max() > threshold:
            EV = EV_new 
            pchoice = self.choice_prob(EV)
            ecost = (pchoice*EV).sum(1)
            futil_maint = np.dot(ecost, self.trans_mat)
            futil_repl = np.dot(ecost, self.regen_mat)
            futil = np.vstack((futil_maint, futil_repl)).T
            
            EV_new = EV_myopic + beta*futil
            k += 1
            if k == 1000:
                achieved = False
                break

        if not suppr_output:
            if achieved:
                print("Convergence achieved in {} iterations".format(k))
            else:
                print("CM could not converge! Mean difference = {:.6f}".format((EV_new-EV).mean()))
                
        self.EV = EV_new
        return EV
        
    def loglike(self, params):
        """
        The log-likelihood of the Dynamic model is estimated in several steps.
        1°) The currenter parameters are supplied to the contraction mapping function
        2°) The function returns a matrix of decision probabilities for each state.
        3°) This matrix is used to compute the loglikelihood of the observations
        4°) The log-likelihood are then summed accross individuals, and returned
        """
        util = self.fl_costs(params, suppr_output=True) 
        pchoice = self.choice_prob(util) 
        logprob = np.log(np.dot(pchoice.T, self.state_mat))
        return -np.sum(self.dec_mat*logprob)
    
    def fit_likelihood(self, x0=None, bounds=None):
        """
        Fit the parameters to the data.
        """
        if bounds == None:
            bounds = [(1e-6, None) for i in range(self.npars)]
            
        if x0 == None:
            x0 = [0.1 for i in range(self.npars)]
            
        self.fitted = opt.fmin_l_bfgs_b(self.loglike, x0=x0, approx_grad=True, 
                                        bounds=bounds)
    
    
    def get_parameters(self):
        """
        Return the parameters obtained after fitting the likelihood function
        to the data.
        """
        return self.fitted[0]
        
    def print_parameters(self):
        loglike =  self.fitted[1]
        fit_params = self.get_parameters()
        RC, thetas = fit_params[0], fit_params[1:]
        logstring = "Log-likelihood = {0:.2f}".format(loglike)
        thetas_string = ["theta1_{0} = {1:.4f}".format(i+1, t) \
                                                for i, t in enumerate(thetas)]
        thetas_string = ", ".join(thetas_string)
        rc_string = "Parameters: RC = {0:.4f}".format(RC)
        print(logstring, rc_string + ", " + thetas_string)

            
          
if __name__ == "__main__":
    import pandas as pd
    
    def lin_cost(s, params):
        theta1_1, = params
        return s*theta1_1
            
    data = pd.read_csv("Lin_Dataset.csv")
    p = np.array([0.36, 0.48, 0.16])
    lin_to_lin = DynamicLogit(data, "Choice", "State", p, lin_cost, npars=3)
    lin_to_lin.fit_likelihood()
    lin_to_lin.print_parameters()