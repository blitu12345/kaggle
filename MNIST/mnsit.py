from numpy import*
import pandas as pd
#import random

class network(object):
	def __init__(self,sizes):
		self.num_layers=len(sizes)
		self.sizes=sizes
		self.biases=[random.randn(y,1) for y in sizes[1:]]
		self.weights=[random.randn(x,y) for x,y in zip(sizes[1:],sizes[:-1])]

	def feedforward(self,a):
		for w,b in zip(self.weights,self.biases):
			a=sigmoid(dot(w,a)+b)
		return a

	def evaluate(self,test_data):
		results =[(argmax(self.feedforward(x)), y) for (x, y) in test_data]
		#print [int(x == y) for (x, y) in results[0]]
		p=sum(int(Yset(y) == x) for (x, y) in results)
		#print shape(x),shape(y)
		#print p
		#for x,y in results:
		#	print Yset(y),x
		#	break
		print str(p)+"/10000"

	"""def evaluate(self,test):
		count=0;p=0
		for t in test[:5]:
			X,Y=t
			maxi=-100;j=0
			f=reshape(self.feedforward(X),(10,))
			#print "f=========",f
			for i in range(10):
				#print f[i]
				#print f
				#print maxi
				if(f[i]>maxi):
					j=i
					maxi=f[i]
			if(j==Y):
				count+=1
		print str(count)+"/10000"""

	def SGD(self,epoch,mini_batch_size,train,eta,test=None):
		n=len(train)
		for i in range(epoch):
			random.shuffle(train)
			mini_batches=[ train[k:k+mini_batch_size] for k in xrange(0,n,mini_batch_size)]
			for mini_batch in mini_batches:
				self.update_mini_batch(mini_batch,eta)
			#print "####################",self.biases
			if test:
				print "epoch"+str(i)+"/"+str(epoch)
				self.evaluate(test)
			else:
				print "epoch"+str(i)

	def update_mini_batch(self,mini_batch,eta):
		delta_b=[zeros(shape(y)) for y in self.biases]
		delta_w=[zeros(shape(y)) for y in self.weights]
		for x,y in mini_batch:
			ndelta_b,ndelta_w=self.backprop(x,y)
			delta_b=[np+bp for np,bp in zip(ndelta_b,delta_b)]
			delta_w=[np+bp for np,bp in zip(ndelta_w,delta_w)]
		#print "previous================",self.biases
		self.weights=[w-(eta/len(mini_batch))*nw  for w,nw in zip(self.weights,delta_w)]
		self.biases=[b-(eta/len(mini_batch))*nb  for b,nb in zip(self.biases,delta_b)]
		#print "after=================",self.biases

	def backprop(self,x,y):
		nabla_b=[zeros(shape(y1)) for y1 in self.biases]
		nabla_w=[zeros(shape(y1)) for y1 in self.weights]
		#print shape(nabla_w)
		#for w in nabla_w:
			#print shape(w)
		#print shape(x),shape(y)
		zs=[];activation=x;activations=[x]
		#print shape(activation)
		for b,w in zip(self.biases,self.weights):
			#print shape(w),shape(activation)
			z=dot(w,activation)+b
			zs.append(z)
			activation=sigmoid(z)
			activations.append(activation)
		delta=Pcost(activation,y)*sigmoid_prime(z)
		nabla_b[-1]=delta
		nabla_w[-1]=dot(delta,activations[-2].transpose())
		#print shape(nabla_w[-1])
		for l in range(2,self.num_layers):
			delta=dot(self.weights[-l+1].transpose(),delta)*sigmoid_prime(zs[-l])
			nabla_b[-l]=delta
			nabla_w[-l]=dot(delta,activations[-l-1].transpose())
		#print shape(nabla_w)
		return nabla_b,nabla_w
def sigmoid(z):
	return(1/(1+exp(-z)))

def sigmoid_prime(z):
	return sigmoid(z)*(1-sigmoid(z))

def Pcost(activation,y):
	return (activation-y)

def Yreset(j):
	y=zeros((10,1))
	y[j]=1.0
	return y
def Yset(arr):
	#print arr
	j=0
	for i in arr:
		if(i==1.0):
			return j
		j+=1

def extract(dataset,matrix=True):
	x1=[reshape(x,(784,1)) for x in dataset[0]]
	if(matrix):
		y1=[Yreset(j) for j in dataset[1]]
	else:
		y1=dataset[1]
	#print shape(x1),shape(y1)
	dataset=zip(x1,y1)
	return dataset

def load():
	import gzip
	import cPickle
	fr=gzip.open('neural-networks-and-deep-learning/data/mnist.pkl.gz','rb')
	train,valid,test=cPickle.load(fr)
	train=extract(train)
	valid=extract(valid,True)
	test=extract(test,True)
	return train,test,valid

#def prediction():



if __name__=="__main__":
	df=pd.read_csv('train.csv')
	y=df['label'].values
	df=df.drop(['label'],axis=1)
	dataset=zip(df.values,y)
	train=extract(dataset,True)
	x,y=train
	print shape(x),shape(y)
	#net=network([784,30,10])
	train,test,valid=load()
	#net.SGD(30,10,train,3.0)
	x,y=train
	print shape(x),shape(y)
