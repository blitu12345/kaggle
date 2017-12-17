import pandas as pd
from numpy import *

df=pd.read_csv('test.csv')
fr=open('genderbasedmodel.csv','w')
m,n=shape(df)
fr.write("PassengerId"+',Survived')
fr.write('\n')
for i  in range(m):
	fr.write(str(i+892)+',')
	if df['Sex'][i] == 'female':
		fr.write('1')
	else:
		fr.write('0')
	fr.write('\n')