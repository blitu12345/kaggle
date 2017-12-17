from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from numpy import *
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.multiclass import OneVsOneClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score

def plot(x,k):
	counter=1
	print "ploting"
	for i in range(1,4):
		for j in range(1,6):
			plt.subplot(3,5,counter)
			plt.imshow(x[(i-1)*10+j].reshape((k,k)),cmap='Greys_r')
			plt.axis('off')
			counter+=1
	plt.show()

def logisticReg(x,y):
	from sklearn.linear_model import LogisticRegression
	from sklearn.cross_validation import train_test_split
	from sklearn.cross_validation import cross_val_score
	from sklearn.metrics import accuracy_score
	x,xt,y,yt=train_test_split(x,y)
	clf=LogisticRegression(x,y)
	clf.fit_transform(x,y)
	xt=clf.transform(xt)
	pred=clf.predict(xt)
	score = accuracy_score(yt,pred)
	print score


def multi(x,y):
	from sklearn.linear_model import LogisticRegression
	from sklearn.cross_validation import train_test_split
	from sklearn.cross_validation import cross_val_score
	from sklearn.metrics import accuracy_score
	from sklearn.multiclass import OneVsRestClassifier
	from sklearn.multiclass import OneVsOneClassifier
	OVR = OneVsRestClassifier(LogisticRegression()).fit(x,y)
	OVO = OneVsOneClassifier(LogisticRegression()).fit(x,y)
	print 'One vs rest accuracy: %.3f' % OVR.score(xt,yt)
	print 'One vs one accuracy: %.3f' % OVO.score(xt,yt)

def go(plot=False):
	df=pd.read_csv('1train.csv')
	y=df['label'].values
	df=df.drop(['label'],axis=1)
	x=df.values
	df1=pd.read_csv('test.csv')
	xt=df1.values
	x=scale(x);xt=scale(xt)
	pca=PCA(n_components=111)
	if(plot):
		var=pca.explained_variance_ratio_
		var1=cumsum(around(var,decimals=4)*100)
		plt.plot(var1)
		plt.show()
	x=pca.fit_transform(x)
	xt=pca.transform(xt)
	return x,y,xt

def linear(x,y,xt):
	OVO = OneVsOneClassifier(LogisticRegression())
	OVO.fit(x,y)
	pred=OVO.predict(xt)
	return pred
def gogo(pred):
	from sklearn.metrics import accuracy_score
	#print accuracy_score(y[2500:],pred)
	fr=open('OvsA.csv','w')
	fr.write('ImageId'+','+'Label')
	fr.write('\n')
	for i in range(len(pred)):
		fr.write(str(i+1)+',')
		fr.write(str(pred[i]))
		fr.write('\n')
	fr.close()

def quadratic(x,xt):
	from sklearn.preprocessing import PolynomialFeatures
	poly = PolynomialFeatures(2)
	x=poly.fit_transform(x)
	xt=poly.transform(xt)
	return x,xt

def knn(x,y,xt,k):
	from sklearn.neighbors import KNeighborsClassifier
	nn=KNeighborsClassifier(n_neighbors=k)
	#x,xt,y,yt=train_test_split(x,y)
	nn.fit(x,y)
	pred=nn.predict(xt)
	#return accuracy_score(yt,pred)
	return pred


if __name__=='__main__':
	x,y,xt=go()
	x,xt=quadratic(x,xt)
	gogo(x,y,xt)