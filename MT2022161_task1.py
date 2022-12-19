STUDENT_NAME = "Aditya.M" #Put your name

STUDENT_ROLLNO = "MT2022161" #Put your roll number

CODE_COMPLETE = True 

# set the above to True if you were able to complete the code

# and that you feel your model can generate a good result

# otherwise keep it as False

# Don't lie about this. This is so that we don't waste time with

# the autograder and just perform a manual check

# If the flag above is True and your code crashes, that's

# an instant deduction of 2 points on the assignment.

#

#@PROTECTED_1_BEGIN

## No code within "PROTECTED" can be modified.

## We expect this part to be VERBATIM.

## IMPORTS 

## No other library imports other than the below are allowed.

## No, not even Scipy

import numpy as np 

import pandas as pd 

import sklearn.model_selection as model_selection 

import sklearn.preprocessing as preprocessing 

import sklearn.metrics as metrics 

from tqdm import tqdm # You can make lovely progress bars using this



## FILE READING: 

## You are not permitted to read any files other than the ones given below.

X_train = pd.read_csv("train_X.csv",index_col=0).to_numpy()

y_train = pd.read_csv("train_y.csv",index_col=0).to_numpy().reshape(-1,)

X_test = pd.read_csv("test_X.csv",index_col=0).to_numpy()

submissions_df = pd.read_csv("sample_submission.csv",index_col=0)

#@PROTECTED_1_END




"""# Pre Processing"""

# one hot encoding the y_train to 10 different columns
# Reference: https://www.delftstack.com/howto/numpy/one-hot-encoding-numpy/
Y_train=np.zeros((y_train.size,y_train.max()+1))
Y_train[np.arange(y_train.size),y_train]=1
Y_train.shape

X_train=X_train/255  # to avoid overflow

X_test=X_test/255   # to avoid overflow

print(X_train.shape)
print(Y_train.shape)

"""# Neural Network class """

# Neural Network Architecture: 784(inputs)x60(hidden layer1)x40(hidden layer 2)x20(hidden layer2)x10(output layer)
class neuralNetwork():
  def __init__(self,learningRate=0.1,iterations=500):
    # Reference : https://www.youtube.com/watch?v=woa34ugDSwY
    # parameters for first(hidden) layer
    self.w1=np.random.rand(60,784)-0.5
    self.b1=np.random.rand(60,1)-0.5

    # parameters for second(hidden) layer
    self.w2=np.random.rand(40,60)-0.5
    self.b2=np.random.rand(40,1)-0.5

    # parameters for third(hidden) layer
    self.w3=np.random.rand(20,40)-0.5
    self.b3=np.random.rand(20,1)-0.5

    #parameters for fourth(output) layer
    self.w4=np.random.rand(10,20)-0.5
    self.b4=np.random.rand(10,1)-0.5

    #setting the learning rate and number of iterations
    self.alpha=learningRate
    self.iterations=iterations

    #other parameters which are used in intenal working of the neural network
    self.s1,self.s2,self.s3,self.s4=None,None,None,None # signal for each layer of neurons respectively
    self.out1,self.out2,self.out3,self.out4=None,None,None,None #output of each layers
    self.dw1,self.dw2,self.dw3,self.dw4=None,None,None,None
    self.db1,self.db2,self.db3,self.db4=None,None,None,None
  
  def ReLU(self,input):
    return np.maximum(input,0)
  
  def tanh(self,input):
    return (np.exp(input)-np.exp(-input))/(np.exp(input)+np.exp(-input))
  
  def sigmoid(self,input):
    return 1.0/(1.0+np.exp(-input))
  
  # Reference:https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
  def softmax(self,input):
    return (np.exp(input)/sum(np.exp(input)))
    
  def feedForward(self,x):

    # for layer 1
    inp1=x.T
    self.s1=self.w1.dot(inp1)+self.b1 # signal 
    self.out1=self.ReLU(self.s1)

    # for layer 2
    inp2=self.out1
    self.s2=self.w2.dot(inp2)+self.b2 #signal
    self.out2=self.ReLU(self.s2)


    # for layer 3
    inp3=self.out2
    self.s3=self.w3.dot(inp3)+self.b3 #signal
    #softmax the final output
    self.out3=self.ReLU(self.s3)

    #for layer 4
    inp4=self.out3
    self.s4=self.w4.dot(inp4)+self.b4 #signal
    #softmax the final output
    self.out4=self.softmax(self.s4)
  
  # Reference:https://stackoverflow.com/questions/46411180/implement-relu-derivative-in-python-numpy
  def der_ReLU(self,input):
    return (input>0)*1
  
  def backPropogate(self,x,y):
    
    # for layer 4[output layer]
    y=y.T
    x=x.T
    m,n=x.shape
    del4=(self.out4-y)
    self.dw4=(1/n)*del4.dot(self.out3.T)
    self.db4=(1/n)*np.sum(del4)

    # for layer 3[hidden layer]
    del3=self.w4.T.dot(del4)*self.der_ReLU(self.s3)
    self.dw3=(1/n)*del3.dot(self.out2.T)
    self.db3=(1/n)*np.sum(del3)

    #for layer 2[hidden layer]
    del2=self.w3.T.dot(del3)*self.der_ReLU(self.s2)
    self.dw2=(1/n)*del2.dot(self.out1.T)
    self.db2=(1/n)*np.sum(del2)
    
    #for layer 1[first layer]
    del1=self.w2.T.dot(del2)*self.der_ReLU(self.s1)
    self.dw1=(1/n)*del1.dot(x.T)
    self.db1=(1/n)*np.sum(del1)

  def updateWeights(self):
    #update all the weights here
    self.w1=self.w1-(self.alpha*self.dw1)
    self.w2=self.w2-(self.alpha*self.dw2)
    self.w3=self.w3-(self.alpha*self.dw3)
    self.w4=self.w4-(self.alpha*self.dw4)
    self.b1=self.b1-(self.alpha*self.db1)
    self.b2=self.b2-(self.alpha*self.db2)
    self.b3=self.b3-(self.alpha*self.db3)
    self.b4=self.b4-(self.alpha*self.db4)

  def testAccuracy(self,y):
    testPred=self.out4.T
    labels=y
    correctPred=0
    totalPred,_=y.shape
    del(_)
    for i in range(totalPred):
      pred=np.argmax(testPred[i])
      label=np.argmax(labels[i])
      if(pred==label):
        correctPred+=1
    return(correctPred/totalPred)
  
  def stochasticGD(self,x,y):
    x,y=x.T,y.T
    for i in tqdm(range(self.iterations)):
      n,m=y.shape
      del(m)
      curIndex=np.random.randint(0,n)
      curX=x[curIndex]
      curY=y[curIndex]
      curX=curX.reshape(784,1)
      curY=curY.reshape(10,1)
      # step 1: forward Propogate 
      self.feedForward(curX)
      # step 2: backward Propogate
      self.backPropogate(curX,curY)
      # step 3: update Weights 
      self.updateWeights()
    x,y=x.T,y.T
    self.feedForward(x)
    acc=self.testAccuracy(y)
    print('Final Accuracy of the model on training data is:',acc*100)

  def batchGD(self,x,y):
    for i in tqdm(range(self.iterations)):
      # step 1: forward Propogate 
      self.feedForward(x)
      # step 2: backward Propogate
      self.backPropogate(x,y)
      # step 3: update Weights 
      self.updateWeights()
      if i%101==0:
        acc=self.testAccuracy(y)
        print('Current Accuracy of the model on training data is:',acc*100)
    acc=self.testAccuracy(y)
    print('Final Accuracy of the model on training data is:',acc*100)
  
  def fit(self,X_train,Y_train,learningRate=None,iterations=None):
    #setting learning rate and number of iterations
    if learningRate!=None:
      self.learningRate=learningRate
    if iterations!=None:
      self.iterations=iterations
      
    #training the model using batch Gradient Descent
    self.batchGD(X_train,Y_train)
  
  def predict(self,x,y=None):
    self.feedForward(x)
    if type(y)==type(x): #i.e yTest value is provided then print the accuracy
      acc=self.testAccuracy(y)
      print('Test Accuracy of the model is:',acc*100)
    return self.out4.T

"""# Validation """

#set validation as False to skip this section
validation=False

if validation==True:
    x=pd.DataFrame(X_train)
    y=pd.DataFrame(Y_train)
    xV,xT,yV,yT=model_selection.train_test_split(x,y,train_size=0.75)
    del(x)
    del(y)
    xV=xV.to_numpy() # xV stands for xValidaton data 
    xT=xT.to_numpy() # xT stands for xTest for validation
    yV=yV.to_numpy()
    yT=yT.to_numpy()
    validationModel=neuralNetwork(learningRate=0.175,iterations=1500)
    validationModel.fit(xV,yV)
    yP=validationModel.predict(xT,yT)

"""# Final Model"""

model=neuralNetwork(learningRate=0.2,iterations=2200)
model.fit(X_train,Y_train)

yPred=model.predict(X_test)

"""# Making Predictions for the test data"""

out=[]
items,_=yPred.shape
for i in range(items):
  out.append(np.argmax(yPred[i]))
del(_)
yPred=pd.DataFrame(out)
del(out)

submissions_df=yPred

#@PROTECTED_2_BEGIN 

##FILE WRITING:

# You are not permitted to write to any file other than the one given below.

submissions_df.to_csv("{}__{}.csv".format(STUDENT_ROLLNO,STUDENT_NAME))

#@PROTECTED_2_END
