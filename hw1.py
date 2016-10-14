from mnist import MNIST
import sklearn.metrics as metrics
import numpy as np

NUM_CLASSES = 10

def load_dataset():
    mndata = MNIST('./data/')
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train/255.0
    X_test = X_test/255.0
    X_train = X_train[:,:,np.newaxis]
    X_test = X_test[:,:,np.newaxis]
    return (X_train, labels_train), (X_test, labels_test)


def train(X_train, y_train, reg=0):
    ''' Build a model from X_train -> y_train '''
    
    '''wT=np.zeros((X_train.shape[1],NUM_CLASSES))'''
    '''m1=np.zeros((1,X_train.shape[1]))
    m2=0'''
    lamda=1
    '''y_train_new=np.zeros((y_train.shape[0],NUM_CLASSES))'''
    y_train_new=one_hot(y_train)
    xT=np.transpose(X_train[:,:,0])
    term1=np.dot(xT,y_train_new)
    term2=lamda*np.eye(X_train.shape[1])+np.dot(xT,X_train[:,:,0])
    w=np.dot(np.linalg.inv(term2),term1)
    '''for index1 in range(NUM_CLASSES):
        for index2 in range(X_train.shape[0]):
            m=np.array(y_train[index2]*X_train[index2,])
            m1=m1+np.transpose(m)
            m2+=np.linalg.norm(X_train[index2,:])**2
        wT[index1,:]=m1/(m2+lamda)
    return np.transpose(wT)'''
    return w
    '''return np.zeros((X_train.shape[0], y_train.shape[0]))'''

def one_hot(labels_train):
    '''Convert categorical labels 0,1,2,....9 to standard basis vectors in R^{10} '''
    a=np.zeros((labels_train.shape[0],NUM_CLASSES));
    count=0;
    for item in labels_train:
        a[count,item]=1
        count+=1
    return a

    '''return np.zeros((X_train.shape[0], NUM_CLASSES))'''

def predict(model, X):
    ''' From model and data points, output prediction vectors '''
    predY=np.dot(X[:,:,0],model)
    predL=np.zeros(X.shape[0])
    for index3 in range(predY.shape[0]):
        predL[index3]=np.argmax(predY[index3,:])
    '''predL=np.argmax(predY)'''
    return predL
    """return np.zeros(X.shape[0])"""

if __name__ == "__main__":
    (X_train, labels_train), (X_test, labels_test) = load_dataset()
    model = train(X_train, labels_train)
    y_train = one_hot(labels_train)
    y_test = one_hot(labels_test)

    pred_labels_train = predict(model, X_train)
    pred_labels_test = predict(model, X_test)


    print("Train accuracy: {0}".format(metrics.accuracy_score(labels_train, pred_labels_train)))
    print("Test accuracy: {0}".format(metrics.accuracy_score(labels_test, pred_labels_test)))
