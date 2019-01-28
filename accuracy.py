import matplotlib.pyplot as plt
import numpy as np

def predPoint(p,a):
    toadd = 0
    if float(p)==a:
        toadd = 1
        #print("A")
    return toadd
    
def predPro(p,a):
    toadd = 0
    #print(p,a)
    if p==a:
        toadd = 0
        #print(p==a)
    else:
        (ph,pa)=(p[1:-1].split(","))
        (rh,ra)=(a[1:-1].split(","))
        #print(ph,pa)
        #print(rh,ra)
        #print("")
        ph=int(ph)
        pa=int(pa)
        rh=int(rh)
        ra=int(ra)
        if (np.sign(ph-pa)==np.sign(rh-ra)):
            toadd = 1
        #print(ph-pa,rh-ra)
    #print(toadd)
    #print("")
    return toadd

def total(season, week, mode, ml):
    classifiers1 = np.zeros((week,2))
    
    
    for a in range(0,week):
        pred=predict(season,a+1,mode,ml)
        #print(pred[4])
        for i in range(0,len(pred[0])):
            t=predPro(pred[2][i],pred[1][i][0])
            classifiers1[a,0]+=t    
        if (a>0):
            classifiers1[a,1]=classifiers1[a-1,1]
        classifiers1[a,1]+=classifiers1[a,0]
    return classifiers1

def graph(data):
    x_axis = list(range(1,len(data)+1))
    names = [["Week","Cumulative"],["b","r"]]
    for i in range(0,2):
        plt.plot(x_axis, data[:,i], names[1][i], label=names[0][i])
    plt.legend(loc='best')
    plt.ylabel('Accuracy %')
    plt.xlabel('Size of training data')
    plt.show()

    print(data[-1])                            
                            

