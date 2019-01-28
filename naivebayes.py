from copy import deepcopy
import numpy as np
import math
import sklearn

class feature:
    parameters = []
    def __init__(self):
        self.parameters = []

    
class binary(feature):
    def setParam(self,x,y,num):
        self.parameters = []
        for i in range(0,num):
            pair = np.concatenate((np.reshape(x,(x.shape[0],1)),np.reshape(y,(y.shape[0],1))),axis=1)
            x_k = pair[pair[:,1] == i][:,np.array([True, False])]
            count = np.bincount(x_k)
            x_0 = count[0]
            if count.size ==1:
                x_1 = 0
            else:
                x_1 =count[1]
            
            if x_0 == 0:
                a = 1/(x_1[0]+2)
                b = (x_1[0]+1)/(x_1[0]+2)
            elif x_1 == 0:
                b = 1/(x_0[0]+2)
                a = (x_0[0]+1)/(x_0[0]+2)
            else:
                a = (x_0[0]+1)/(x_0[0]+x_1[0]+2)
                b = (x_1[0]+1)/(x_0[0]+x_1[0]+2)    
            self.parameters.append((a,b))
    def predicts(self,x,C):
        p_0,p_1 = self.parameters[C]
        if x == 0:
            p = p_0**(1-x) * (1-p_0) **(x)
        elif x == 1:
            p = p_1**x * (1-p_1) **(1-x)
        return p
    
class real(feature):
    def setParam(self,x,y,num): 
        self.parameters = []
        for i in range(0,num):
            #print(np.reshape(x,(x.shape[0],1)))
            #print(y)
            pair = np.concatenate((np.reshape(x,(x.shape[0],1)),np.reshape(y,(y.shape[0],1))),axis=1)
            #print(pair[:,1].dtype)
            #print(np.where(pair[:,1] == i))
            #print(pair[pair[:,1]==i])
            #print(pair.shape)
            x_k = pair[pair[:,1] == i][:,np.array([True, False])]
            if (x_k.shape[0]==0):
                M=0 
                V=0
            else:
                M = np.sum(x_k)/x_k.shape[0]
                V = np.sum(np.square(x_k - M))/x_k.shape[0]
            if V == 0:
              V = 10**-6
            self.parameters.append((M,V))
            
            
    def predicts(self,x,C):
        m,v = self.parameters[C]
        
        p = 1/(math.sqrt(2*math.pi*v)) * math.exp((-((float(x)-m)**2)/v))
        return p
        print (p)
            
class NBC:
    features = []
    classes = []
    num_classes = 0
    train_size = 0
    class_name = []
    def __init__(self, feature_types, num_classes, class_name):
        self.features = []
        for f in feature_types:
            if f == 'r':
                i = real()
            elif f == 'b':
                i = binary()
            else:
                i = feature()
            self.features.append(i)
            
        self.num_classes = num_classes
        self.class_name = class_name
    def fit(self,X,y):
        y_ = np.array([(lambda x:self.class_name.index(x))(x) for x in y])
        i = -1
        for f in self.features:
            i += 1
            f.setParam(X[:,i],y_,self.num_classes)
        #y_c = stats.itemfreq(y)
        y_n = np.bincount(y_)
        #print(y_n)
        self.classes = []
        for c in range(0,self.num_classes):
            if c < y_n.size:
                self.classes.append(y_n[c]-1)
            else:
                self.classes.append(0)
        self.train_size = y.size
    def predict(self,X):
        y_hat = []
        #print(y_hat.dtype)
        for i in range(0,X.shape[0]):
            
            C_k = []
            for j in range(0,self.num_classes):
                p = 1.0
                k = -1
                for f in self.features:
                    k += 1
                    p = p*f.predicts(X[i,k],j)
                    #print(p)
                    #print("")
                C_k.append(p*self.classes[j]/self.train_size)
            y_hat.append(self.class_name[np.argmax(C_k)])
            #print(np.argmax(C_k))
            #print(self.class_name[np.argmax(C_k)])
            #print(y_hat[i])
            #print("")
        return y_hat


