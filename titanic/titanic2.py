import pandas as pd
from numpy import*

def traindata():
	df=pd.read_csv('train.csv')

	df['Gender']=0
	df['Gender']=df['Sex'].map({'female':0,'male':1}).astype(int)

	df['Nage']=4
	df['Nage']=df['Age']

	df['From']=df['Embarked'].map({'C':1,'Q':2,'S':3}).astype(float)
	df['from1']=df['From']
	median_from=zeros((2,3))
	median_ages=zeros((2,3))
	for i in range(2):
		for j in range(1,4):
			median_ages[i,j-1]=df[(df['Gender']==i) & (df['Pclass']==j)]['Age'].dropna().median()
			median_from[i,j-1]=df[(df['Gender']==i) & (df['Pclass']==j)]['from1'].dropna().median()

	for i in range(2):
		for j in range(1,4):
			df.loc[ (df.Age.isnull()) & (df.Gender==i) & (df.Pclass==j),'Nage']=median_ages[i,j-1]
			df.loc[(df.from1.isnull()) & (df.Gender==i) & (df.Pclass==j),'From']=median_ages[i,j-1]
	df['Nfare']=df['Fare']
#	df['Nfare']=(df['Nfare']-df['Nfare'].mean())/(df['Nfare'].max() -df['Nfare'].min())
	df['Relative']=df['Parch']+df['SibSp']
	df=df.drop(['PassengerId','Name','Age','SibSp','Parch','Ticket','Cabin','Embarked','Sex','Fare','from1'],axis=1)
	#eturn df
	return df.values

def testdata():
	df=pd.read_csv('test.csv')

	df['Gender']=0
	df['Gender']=df['Sex'].map({'female':0,'male':1}).astype(int)

	df['Nage']=4
	df['Nage']=df['Age']

	median_ages=zeros((2,3))
	for i in range(2):
		for j in range(1,4):
			median_ages[i,j-1]=df[(df['Gender']==i) & (df['Pclass']==j)]['Age'].dropna().median()

	for i in range(2):
		for j in range(1,4):
			df.loc[ (df.Age.isnull()) & (df.Gender==i) & (df.Pclass==j),'Nage']=median_ages[i,j-1]

	df['From']=df['Embarked'].map({'C':1,'Q':2,'S':3}).astype(float)
	df['from1']=df['From']
	median_from=zeros((2,3))

	df['Nfare']=df['Fare']
	median_ages=zeros((2,3))
	for i in range(2):
		for j in range(1,4):
			median_ages[i,j-1]=df[(df['Gender']==i) & (df['Pclass']==j)]['Fare'].dropna().median()
			median_from[i,j-1]=df[(df['Gender']==i) & (df['Pclass']==j)]['from1'].dropna().median()

	for i in range(2):
		for j in range(1,4):
			df.loc[(df.Fare.isnull()) & (df.Gender==i) & (df.Pclass==j),'Nfare']=median_ages[i,j-1]
			df.loc[(df.from1.isnull()) & (df.Gender==i) & (df.Pclass==j),'From']=median_ages[i,j-1]

	df['Relative']=df['Parch']+df['SibSp']
#	df['Nfare']=(df['Nfare']-df['Nfare'].mean())/(df['Nfare'].max() -df['Nfare'].min())
	df=df.drop(['PassengerId','Name','Age','SibSp','Parch','Ticket','Cabin','Embarked','Sex','Fare','from1'],axis=1)
	print df.info()
	return df.values

def RandomForest(x_train,y_train,x_test):
	from sklearn.feature_selection import SelectFromModel
	fr=open('random_forest.csv','w')
	fr.write('PassengerId'+','+'Survived')
	fr.write('\n')
	from sklearn.ensemble import RandomForestClassifier
	forest=RandomForestClassifier()
	forest=forest.fit(x_train,y_train)
	#feat=pd.DataFrame()
	#feat['features']=df.columns
	#feat['imp']=forest.feature_importances_
	#feat.sort(['imp'],ascending=False)
	#model=SelectFromModel(forest,prefit=True)
	#x_train=model.transform(x_train)
	#x_test=model.transform(x_test)
	forest=RandomForestClassifier(n_estimators=210,criterion='gini',max_depth=7)
	forest=forest.fit(x_train,y_train)
	pred=forest.predict(x_test).astype(int)
	index=892
	for i in pred:
		fr.write(str(index))
		fr.write(',')
		fr.write(str(i))
		fr.write('\n')
		index+=1
	fr.close()
	#return x_train

def svmClassifier(x_train,y_train,x_test):
	from sklearn import svm
	from sklearn import preprocessing
	fr=open('svm.csv','w')
	fr.write('PassengerId'+','+'Survived')
	fr.write('\n')
	x_train_scaled=preprocessing.scale(x_train)
	x_test_scaled=preprocessing.scale(x_test)
	clf=svm.SVC(gamma=0.01,C=30)
	clf.fit(x_train_scaled,y_train)
	pred=clf.predict(x_test_scaled).astype(int)
	index=892
	for i in pred:
		fr.write(str(index))
		fr.write(',')
		fr.write(str(i))
		fr.write('\n')
		index+=1
	fr.close()


if __name__=='__main__':
	trainData=traindata()
	x_train,y_train=trainData[0::,1::],trainData[0::,0]
	x_test=testdata()
	svmClassifier(x_train,y_train,x_test)
	#RandomForest(x_train,y_train,x_test)