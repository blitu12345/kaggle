import pandas as pd
from numpy import*
import pylab as plt

def visualize():
	df=pd.read_csv("train.csv")
	survived=df.groupby('Embarked')['Survived'].sum()
	total=df.groupby('Embarked')['PassengerId'].count()
	totall=total.sum()
	fig=plt.figure()
	ax=fig.add_subplot(111)
	bar_width=0.35
	index=arange(len(survived.index.values))
	rect1=ax.bar(index,survived/totall,bar_width,color='blue',label='survived')
	rect2=ax.bar(index+bar_width,total/totall,bar_width,color='green',label='total passenger')
	ax.set_xticks(array(index)+bar_width)
	XticksMark=[1,2,3]
	XtickMarkLabel=ax.set_xticklabels(XticksMark)
	plt.setp(XtickMarkLabel,fontsize=20)
	plt.show()


if __name__=="__main__":
	visualize()