from sklearn.base import BaseEstimator
import numpy as np
import scipy.stats as stats

# For this assignment we will implement the Naive Bayes classifier as a
# a class, sklearn style. You only need to modify the fit and predict functions.
# Additionally, implement the Disparate Impact measure as the evaluateBias function.
class NBC(BaseEstimator):
    '''
    (a,b) - Beta prior parameters for the class random variable
    alpha - Symmetric Dirichlet parameter for the features
    '''

    def __init__(self, a=1, b=1, alpha=1):
        self.a = a
        self.b = b
        self.alpha = alpha
        self.params = None
        
    def get_a(self):
        return self.a

    def get_b(self):
        return self.b

    def get_alpha(self):
        return self.alpha

    # you need to implement this function

    def fit(self,X,y):
        '''
        This function does not return anything
        
        Inputs:
        X: Training data set (N x d numpy array)
        y: Labels (N length numpy array)
        '''
        
        a = self.get_a()
        b = self.get_b()
        alpha = self.get_alpha()
        self.classes = np.unique(y)
        
        #print(len(X[:]))
        # remove next line and implement from here
        # you are free to use any data structure for paramse
        def classCondProb(j, lab):
            retVal = {}
            un = np.unique(j)
            kj =  len(un)
            n = np.count_nonzero(y == lab)
            
            for val in un:
                nj = 0
                for i in range(len(y)):
                    if j[i] == val and y[i] == lab:
                        nj = nj + 1
                
                retVal[val] = (nj + alpha)/(n + kj*alpha)
            retVal['null'] = (alpha)/(n + kj*alpha)
            return retVal

        
            
        N = len(y)
        N1 = np.count_nonzero(y == 1)
       
        theta1J =  []
        theta2J = []
        for v in range(len(X[0])):
            theta1J.append(classCondProb(X[:,v], 1))
            theta2J.append(classCondProb(X[:,v], 2))  
        #print(b)
        thetaB = (N1 + a)/(N + a + b)
         

        params = [thetaB, theta1J, theta2J] 
        #print(params)
        # do not change the line below
        self.params = params
    
    # you need to implement this function
    def predict(self,Xtest):
        '''
        This function returns the predicted class for a given data set
        
        Inputs:
        Xtest: Testing data set (N x d numpy array)
        
        Output:
        predictions: N length numpy array containing the predictions
        '''
        params = self.params
        a = self.get_a()
        b = self.get_b()
        alpha = self.get_alpha()
        
        #remove next line and implement from here
        predictions = []

        def prob(p, x1, ind):
            
            feats = Xtest[ind]
            retVal = 1
        
            for i in range(len(feats)):
                di = x1[i]
                if feats[i] in di:
                    retVal = retVal*di[feats[i]]
                else:
                    retVal = retVal*di['null']
            return retVal*p

            


            
        for x in range(len(Xtest[:])):
            p1 = prob(params[0], params[1], x)
            p2 = prob(1-params[0], params[2], x)
            P1 = p1/(p1+p2)
            P2 = p2/(p1+p2)
            
            if P1 >= P2:
                predictions.append(1)
            else:
                predictions.append(2)

        #predictions = np.random.choice(self.__classes,np.unique(Xtest.shape[0]))
        #do not change the line below
        #print(predictions)
        return predictions
        
def evaluateBias(y_pred,y_sensitive):
    '''
    This function computes the Disparate Impact in the classification predictions (y_pred),
    with respect to a sensitive feature (y_sensitive).
    
    Inputs:
    y_pred: N length numpy array
    y_sensitive: N length numpy array
    
    Output:
    di (disparateimpact): scalar value
    '''
    #remove next line and implement from here
    y1 = 0
    s1 = 0
    y2 =0 
    s2 = 0
    di = 0
    for i in range(len(y_pred)):
        if y_sensitive[i] == 1:
            s1 = s1 + 1
            if y_pred[i] == 2:
                y1 = y1 + 1
        elif  y_sensitive[i] == 2:
            s2 = s2 + 1
            if y_pred[i] == 2:
                y2 = y2 + 1
    
    di = (y2/s2)/(y1/s1)
    
    #do not change the line below
    return di

def genBiasedSample(X,y,s,p,nsamples=1000):
    '''
    Oversamples instances belonging to the sensitive feature value (s != 1)
    
    Inputs:
    X - Data
    y - labels
    s - sensitive attribute
    p - probability of sampling unprivileged customer
    nsamples - size of the resulting data set (2*nsamples)
    
    Output:
    X_sample,y_sample,s_sample
    '''
    i1 = y == 1 # good
    i1 = i1[:,np.newaxis]
    i2 = y == 2 # bad
    i2 = i2[:,np.newaxis]
    
    sp = s == 1 #privileged
    sp = sp[:,np.newaxis]
    su = s != 1 #unprivileged
    su = su[:,np.newaxis]

    su1 = np.where(np.all(np.hstack([su,i1]),axis=1))[0]
    su2 = np.where(np.all(np.hstack([su,i2]),axis=1))[0]
    sp1 = np.where(np.all(np.hstack([sp,i1]),axis=1))[0]
    sp2 = np.where(np.all(np.hstack([sp,i2]),axis=1))[0]
    inds = []
    for i in range(nsamples):
        u = stats.bernoulli(p).rvs(1)
        if u == 1:
            #sample one bad instance with s != 1
            inds.append(np.random.choice(su2,1)[0])
            #sample one good instance with s == 1
            inds.append(np.random.choice(sp1,1)[0])
        else:
            #sample one good instance with s != 1
            inds.append(np.random.choice(su1,1)[0])
            #sample one bad instance with s == 1
            inds.append(np.random.choice(sp2,1)[0])
    X_sample = X[inds,:]
    y_sample = y[inds]
    s_sample = s[inds]
    
    return X_sample,y_sample,s_sample,inds
