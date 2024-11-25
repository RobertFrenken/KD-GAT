#recursive least squares --adam dhalla
import numpy as np 
import numpy.linalg 

class RecursiveLeastSquares():
    """Creates a RecursiveLeastSquares object, that will efficiently returned modified x's 
       as more data is inputted using the .addData() method. Updates to x are calculated using
       the Sherman-Morrison-Woodburry Formula. 

       - initA (Ndarray)
                the initial A matrix in least squares, before adding any data. The calculations 
                will be based off this initial matrix. Is size (examples, variables). Think of 
                the A matrix in the Normal Equations. 

       - initB (Ndarray)
               the initial "answers" B matrix. Same B in the Normal Equations. Size (examples, 1)
    """
    def __init__(self, initA, initb):
        self.A = initA  
        self.b = initb 

        # create the initial P matrix, the (A^T*A)^-1 matrix.
        # we don't link it to self.A, self.b since these will change, and after
        # the first P, we will use S-M-W to calculate P instead.
        
        if initA.size < 2:
            initialP = [[1]]
            self.P = [[1]]
        else: 

            initialP = np.linalg.inv(np.dot(initA.transpose(), initA))
            self.P = initialP

        # do least squares automatically the first round 

        # self.K is the other part of the normal equation that multiplies P, (A^T)*B
        self.K = None

        # do least squares automatically for first time
        if initA.size != 0:
            self.x = np.dot(initialP, np.dot(initA.transpose(), initb))
        else:
            self.x = [[1]] 
    
    def addData(self, newA, newb):
        """add data to the least squares problems and returns an updated x.
           
           - newA    (ndarray)
                     adding more rows to the A matrix. Often a row vector (if adding one 
                     more data point). Otherwise, size is (newpoints, variables)
             
           - newb    (ndarray)
                     adds corresponding 'output' for the newA. A (1, 1) ndarray if adding
                     only one more data point. Else, size is (newpoints, 1)

            Returns the updated x. 
        """ 

        newA = newA.reshape(-1, (np.shape(newA)[0]))
        self.A = np.concatenate([self.A, newA])

        newb = newb.reshape(-1, 1)
        self.b = np.concatenate([self.b, newb])

        # create P by using Sherman-Morrison-Woodburry
        # I separate the formula into chunks for readability, see README for details

        # size of I depends on rows of data inputted
        I = (np.eye(np.shape(newA)[0]))


        PIn = np.linalg.inv(I + np.dot(newA, np.dot(self.P, newA.transpose())))
        PinA = np.dot(np.dot(newA.transpose(), PIn), newA)
        PinAP = np.dot(np.dot(self.P, PinA), self.P)
        P = self.P - PinAP

        # create K 
        self.P = P
        self.K = np.dot(self.P, newA.transpose())


        Q = newb - np.dot(newA, self.x)

        self.x = self.x + np.dot(self.K, Q)
        return self.x
    
"""
This is an implementation of the recursive least-squares method that is derived and explained here

https://aleksandarhaber.com/introduction-to-kalman-filter-derivation-of-the-recursive-least-squares-method-with-python-codes/

Author: Aleksandar Haber 
Last Revision: October 25, 2022

"""

class RecursiveLeastSquares2(object):
    
    # x0 - initial estimate used to initialize the estimator
    # P0 - initial estimation error covariance matrix
    # R  - covariance matrix of the measurement noise
    def __init__(self,x0,P0,R):
        
        # initialize the values
        self.x0=x0
        self.P0=P0
        self.R=R
        
        # this variable is used to track the current time step k of the estimator 
        # after every time step arrives, this variables increases for one 
        # in this way, we can track the number of variblaes
        self.currentTimeStep=0
                  
        # this list is used to store the estimates xk starting from the initial estimate 
        self.estimates=[]
        self.estimates.append(x0)
         
        # this list is used to store the estimation error covariance matrices Pk
        self.estimationErrorCovarianceMatrices=[]
        self.estimationErrorCovarianceMatrices.append(P0)
        
        # this list is used to store the gain matrices Kk
        self.gainMatrices=[]
         
        # this list is used to store estimation error vectors
        self.errors=[]
    
     
    # this function takes the current measurement and the current measurement matrix C
    # and computes the estimation error covariance matrix, updates the estimate, 
    # computes the gain matrix, and the estimation error
    # it fills the lists self.estimates, self.estimationErrorCovarianceMatrices, self.gainMatrices, and self.errors
    # it also increments the variable currentTimeStep for 1
    
    # measurementValue - measurement obtained at the time instant k
    # C - measurement matrix at the time instant k
    
    def predict(self,measurementValue,C):
        import numpy as np
        
        # compute the L matrix and its inverse, see Eq. 43
        Lmatrix=self.R+np.matmul(C,np.matmul(self.estimationErrorCovarianceMatrices[self.currentTimeStep],C.T))
        LmatrixInv=np.linalg.inv(Lmatrix)
        # compute the gain matrix, see Eq. 42 or Eq. 48
        gainMatrix=np.matmul(self.estimationErrorCovarianceMatrices[self.currentTimeStep],np.matmul(C.T,LmatrixInv))

        # compute the estimation error                    
        error=measurementValue-np.matmul(C,self.estimates[self.currentTimeStep])
        # compute the estimate, see Eq. 49
        estimate=self.estimates[self.currentTimeStep]+np.matmul(gainMatrix,error)
        
        # propagate the estimation error covariance matrix, see Eq. 50            
        ImKc=np.eye(np.size(self.x0),np.size(self.x0))-np.matmul(gainMatrix,C)
        estimationErrorCovarianceMatrix=np.matmul(ImKc,self.estimationErrorCovarianceMatrices[self.currentTimeStep])
        
        # add computed elements to the list
        self.estimates.append(estimate)
        self.estimationErrorCovarianceMatrices.append(estimationErrorCovarianceMatrix)
        self.gainMatrices.append(gainMatrix)
        self.errors.append(error)
        
        # increment the current time step
        self.currentTimeStep=self.currentTimeStep+1