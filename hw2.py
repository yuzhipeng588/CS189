from mnist import MNIST
import sklearn.metrics as metrics
import numpy as np
import scipy
import matplotlib.pyplot as plt

NUM_CLASSES = 10

def load_dataset():
    mndata = MNIST('./data/')
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train/255.0
    X_test = X_test/255.0
    return (X_train, labels_train), (X_test, labels_test)


def train(X_train, y_train, reg=0):
    ''' Build a model from X_train -> y_train '''
    lamda=0.3
    '''y_train_new=one_hot(y_train)'''
    xT=np.transpose(X_train)
    term1=np.dot(xT,y_train)
    norm=np.dot(xT,X_train)
    term2=lamda*np.eye(X_train.shape[1])+norm
    w=np.dot(np.linalg.inv(term2),term1)
    return w
    '''return np.zeros((X_train.shape[1], NUM_CLASSES))'''

def train_gd(X_train, y_train,modelclosed,alpha=0.1, reg=0, num_iter=10000):
    ''' Build a model from X_train -> y_train using batch gradient descent '''
    '''w=wl=modelclosed'''
    '''w=train(X_train, y_train, reg=0)'''
 
    ''' 
    scipy.optimize.fmin_bfgs
    '''     
    w0=scipy.zeros(X_train.shape[1]*NUM_CLASSES)
    '''w0=modelclosed'''
    def func(params, *args):
        w=params
        X_train=args[0]
        y_train=args[1]
        wf=w.reshape(X_train.shape[1],NUM_CLASSES)
        return 0.1*scipy.linalg.norm(wf)+scipy.linalg.norm(scipy.dot(X_train,wf)-y_train)
    def grad(params,*args):
        w=params
        X_train=args[0]
        y_train=args[1]
        wf=w.reshape(X_train.shape[1],NUM_CLASSES)
        y_pred=np.dot(X_train,wf)
        loss=y_pred-y_train
        gdmatrix=scipy.dot(scipy.transpose(X_train),loss)/X_train.shape[0]
        '''gdarray=np.squeeze(np.asarray(gdmatrix))'''
        gdarray=gdmatrix.flatten()
        return gdarray
    wmin=scipy.optimize.fmin_bfgs(func,x0=w0,fprime=grad,args=(X_train,y_train))
    '''wmin=scipy.optimize.fmin_bfgs(func,x0=w0,args=(X_train,y_train))'''
    
    '''Normal GD in for loop '''    
    '''
    w=wl=np.zeros((X_train.shape[1], NUM_CLASSES))
    Terros=[]
    lamda=1
    for i in range(100):    
        y_pred=np.dot(X_train,w)
        Terros.append(TrainingErrors(y_pred))
        loss=y_pred-y_train
        gradient=np.dot(np.transpose(X_train),loss)/X_train.shape[0]+2*lamda*w
        wn=w-10*alpha*gradient+0.9*(w-wl)
        wl=w
        w=wn
    plt.plot(Terros)
    '''
    '''y_pred=np.dot(X_train,w)
    y_TF=y_pred-y_train
    indexY=np.nonzero(y_TF)
    indexY_0=indexY[0]
    for i in range(10):
        y_pred=np.dot(X_train,w)
        y_TF=y_pred-y_train
        indexY=np.nonzero(y_TF)
        indexY_0=indexY[0]
        if(indexY_0.shape[0]<100):
            break
        for index in indexY_0:
            A=np.mat(X_train[index,:])
            A=np.transpose(A)
            B=np.mat(y_train[index,:])
            w+=alpha*np.dot(A,B)'''
    '''return np.zeros((X_train.shape[1], NUM_CLASSES))'''
        
    return wmin.reshape(X_train.shape[1],NUM_CLASSES)

def train_sgd(X_train, y_train, alpha=0.1, reg=0, num_iter=10000):
    ''' Build a model from X_train -> y_train using stochastic gradient descent '''
    w=np.zeros((X_train.shape[1], NUM_CLASSES))
    Terros=[]
    '''w=train(X_train, y_train, reg=0)'''
    for i in range(100000):    
        y_pred=np.dot(X_train,w)
        '''Terros.append(TrainingErrors(y_pred))'''
        loss=y_pred-y_train
        ran=np.random.randint(0,59999)
        Xi=np.mat(X_train[ran,:])
        XiT=np.mat(np.transpose(Xi))
        Lossi=np.mat(loss[ran,:])
        '''gradient=np.dot(np.linalg.inv(np.dot(np.transpose(X_train),X_train)),np.dot(XiT,Lossi))'''
        gradient=np.dot(XiT,Lossi)
        '''gradient=np.dot(np.linalg.inv(np.dot(np.transpose(X_train[ran,:]),X_train[ran,:])),np.dot(np.transpose(X_train[ran,:]),loss[ran,:]))'''
        w=w-alpha*gradient
    '''return np.zeros((X_train.shape[1], NUM_CLASSES))'''
    '''plt.plot(Terros) '''   
    return w

def one_hot(labels_train):
    '''Convert categorical labels 0,1,2,....9 to standard basis vectors in R^{10} '''
    a=np.zeros((labels_train.shape[0],NUM_CLASSES));
    count=0;
    for item in labels_train:
        a[count,np.int_(item)]=1
        count+=1
    return a
    '''return np.zeros((X_train.shape[0], NUM_CLASSES))'''

def predict(model, X):
    ''' From model and data points, output prediction vectors '''
    predY=np.dot(X,model)
    predL=np.zeros(X.shape[0])
    for index3 in range(predY.shape[0]):
        '''P=np.abs(predY[index3,:]-1)'''
        predL[index3]=np.argmax(predY[index3,:])
    '''predL=np.argmax(predY)'''
    return predL
    '''return np.zeros(X.shape[0])'''

def phi(X,wr,b):
    ''' Featurize the inputs using random Fourier features '''
    '''Id=np.eye(X.shape[1])
    wr=np.random.multivariate_normal(np.zeros(X.shape[1]),100**2*Id,700)
    b=2*np.pi*np.random.random(700)'''
    '''X=np.concatenate((X, np.sqrt(2)*np.dot(X,np.transpose(wr))+np.transpose(b)), axis=1)'''
    X=np.sqrt(2.0/wr.shape[0])*np.cos(np.dot(X,np.transpose(wr))+np.transpose(b))
    return X
def TrainingErrors(Ypred):
    predL=np.zeros(Ypred.shape[0])
    for index3 in range(Ypred.shape[0]):
        '''P=np.abs(Ypred[index3,:]-1)'''
        predL[index3]=np.argmax(Ypred[index3,:])
    return metrics.accuracy_score(labels_train, predL)

if __name__ == "__main__":
    (X_train, labels_train), (X_test, labels_test) = load_dataset()
    ''' '''
    Id=np.eye(X_train.shape[1])
    wr=np.random.multivariate_normal(np.zeros(X_train.shape[1]),0.2**2*Id,5000)
    b=2*np.pi*np.random.random(5000)
    ''' '''
    y_train = one_hot(labels_train)
    y_test = one_hot(labels_test)
    X_train, X_test = phi(X_train,wr,b), phi(X_test,wr,b)
    
    modelclosed = train(X_train, y_train, reg=0.1)
    pred_labels_train = predict(modelclosed, X_train)
    pred_labels_test = predict(modelclosed, X_test)
    print("Closed form solution")
    print("Train accuracy: {0}".format(metrics.accuracy_score(labels_train, pred_labels_train)))
    print("Test accuracy: {0}".format(metrics.accuracy_score(labels_test, pred_labels_test)))
    


    modelgd = train_gd(X_train, y_train,modelclosed, alpha=1e-3, reg=0.1, num_iter=20000)
    pred_labels_train = predict(modelgd, X_train)
    pred_labels_test = predict(modelgd, X_test)
    print("Batch gradient descent")
    print("Train accuracy: {0}".format(metrics.accuracy_score(labels_train, pred_labels_train)))
    print("Test accuracy: {0}".format(metrics.accuracy_score(labels_test, pred_labels_test)))
'''
    model = train_sgd(X_train, y_train, alpha=1e-3, reg=0.1, num_iter=100000)
    pred_labels_train = predict(model, X_train)
    pred_labels_test = predict(model, X_test)
    print("Stochastic gradient descent")
    print("Train accuracy: {0}".format(metrics.accuracy_score(labels_train, pred_labels_train)))
    print("Test accuracy: {0}".format(metrics.accuracy_score(labels_test, pred_labels_test)))
'''