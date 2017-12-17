import pandas as pd
from numpy import*
from titanic2 import*

def combine():
	df_train=pd.read_csv('train.csv')
	df_test=pd.read_csv('test.csv')
	df_train=df_train.drop(['Survived'],axis=1)
	combined=df_train.append(df_test)
	combined.reset_index(inplace=True)
	return combined

def name():
	combined=combine()
	combined['Title']=combined.Name.map(lambda name: name.split(',')[1].split('.')[0].strip())
	title_dict={"Capt":"Officer",
				"Sir":"Officer",
				"Col": "Officer",
				"Major":"Officer",
				"Jonkheer":"Royality",
				"Don": "Royality",
				"Sir":"Royality",
				"Dr":"Officer",
				"Rev":"Officer",
				"the Countess":"Royality",
				"Dona":"Royality",
				"Mme":"Mrs",
				"Mlle":"Miss",
				"Ms":"Mrs",
				"Mr":"Mr",
				"Mrs":"Mrs",
				"Miss":"Mrs",
				"Master":"Master",
				"Lady":"Royality"
				}
	combined['Title']=combined.Title.map(title_dict)
	dummy_titles=pd.get_dummies(combined['Title'],prefix='Title')
	combined=pd.concat([combined,dummy_titles],axis=1)
	combined=combined.drop(['Name','Title'],axis=1)
	return combined

def gender():
	combined=name()
	combined['Gender']=combined['Sex'].map({'male':0,'female':1}).astype(int)
	combined=combined.drop(['Sex'],axis=1)
	return combined

def age():
	df=gender()
	median_age=zeros((2,3))
	for i in range(2):
		for j in range(1,4):
			median_age[i][j-1]=df[ (df['Gender']==i) & (df['Pclass']==j) ]['Age'].dropna().median()

	for i in range(2):
		for j in range(1,4):
			df.loc[ (df['Age'].isnull()) & (df['Gender']==i) & (df['Pclass']==j),'Age']=median_age[i][j-1]
	return df

def age1():
	df1=gender()
	df=df1
	df=df.drop(['Ticket','Cabin','Embarked','Fare'],axis=1)
	from sklearn.linear_model import LinearRegression
	clf=LinearRegression()
	labels=df[df['Age'].notnull()]['Age'].values
	dftest=df[df['Age'].isnull()]
	dftrain=df[df['Age'].notnull()]
	dftest=dftest.drop(['Age'],axis=1)
	test=dftest.values
	dftrain=dftrain.drop(['Age'],axis=1)
	#return dftrain
	train=dftrain.values
	#return train,labels,test
	clf.fit(train,labels)
	pred=clf.predict(test).astype(int)
	df1['Age'].fillna(pred,inplace=True)
	return df1

def fare():
	combined=age1()
	combined['Fare'].fillna(combined['Fare'].mean(),inplace=True)
	return combined

def embarked():
	combined=fare()
	dummy_embarke=pd.get_dummies(combined['Embarked'],prefix='Embarke')
	combined=pd.concat([combined,dummy_embarke],axis=1)
	combined=combined.drop(['Embarked'],axis=1)
	return combined

def pclass():
	combined=embarked()
	dummy_class=pd.get_dummies(combined['Pclass'],prefix='Pclass')
	combined=pd.concat([combined,dummy_class],axis=1)
	combined=combined.drop(['Pclass'],axis=1)
	return combined

def family():
	combined=pclass()
	combined['family']=combined['SibSp'] + combined['Parch']+1
	combined['Singleton']=combined['family'].map(lambda s:1 if s==1 else 0)
	combined['Small']=combined['family'].map(lambda s:1 if 2<=s<=4 else 0)
	combined['Large']=combined['family'].map(lambda s:1 if s>=5 else 0)
	combined=combined.drop(['SibSp','Parch','family','index','Cabin'],axis=1)
	return combined

def ticket():
	df=family()
	#tket=df['Ticket'].replace(to_replace='.',value='')
	tket=df['Ticket'].replace({'/':''},regex=True)
	tket=map(lambda t: t.replace('.',''),tket)
	#tket=tket.replace({".":"p"},regex=True)
	#tket=tket.split()
	tket=map(lambda t: t.split(),tket)
	#tket=map(lambda t: t.strip(),tket)
	tket=map(lambda t:"XXX" if len(t)==0 or t[0].isdigit() else t[0],tket)
	df1=pd.Series(tket)
	dummy_ticket=pd.get_dummies(df1,prefix='Ticket')
	df=pd.concat([df,dummy_ticket],axis=1)
	df=df.drop(['Ticket'],axis=1)
	return df


def cabin():
	df=ticket()
	df['Cabin'].fillna('X',inplace=True)
	cbn=df['Cabin'].values.astype(str)
	cbn=map(lambda t: list(t)[0],cbn)
	df1=pd.Series(cbn)
	dummy_cabin=pd.get_dummies(df1,prefix='Cabin')
	df=pd.concat([df,dummy_cabin],axis=1)
	df=df.drop(['Cabin'],axis=1)
	return df

def recover():
	df=ticket()
	label=pd.read_csv('train.csv').Survived#.astype(float)
	train=df.ix[0:890]#.values.astype(float)
	test=df.ix[891:]#.astype(float)
	return train,label,test

def selection():
	from sklearn.ensemble import ExtraTreesClassifier
	from sklearn.feature_selection import SelectFromModel
	train,label,test=recover()
	clf=ExtraTreesClassifier(n_estimators=200)
	clf.fit(train,label)

	features=pd.DataFrame()
	features['feature']=train.columns
	features['importance']=clf.feature_importances_
	features.sort(['importance'],ascending=False)
	return features

def gridSearch():
	x,y,xt=recover()
	from sklearn.grid_search import GridSearchCV
	from sklearn.ensemble import RandomForestClassifier
	clf=RandomForestClassifier()
	parameters={'max_depth':[4,5,6,7,8],
				'n_estimators':[200,210,240,250],
				'criterion':['gini','entropy']}
	grid_search=GridSearchCV(clf,param_grid=parameters)
	grid_search.fit(x,y)
	print grid_search.best_params_


def classifier0():
	x,y,xt=recover()
	RandomForest(x,y,xt)
	#return b
	#svmClassifier(x.values,y.values,xt.values)