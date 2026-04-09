import pandas as pd 
import joblib
from sklearn.linear_model import LogisticRegression

# ENTRAINER UNE REGRESSION LOGISTIQUE AVEC SIKITLEARN
data = pd.read_csv('data/customer_churn.csv')
x = data[['Age','Account_Manager','Years','Num_Sites']]
 #x=data.drop('Churn',axis=1)
y=data['Churn']
#from sklearn.model_selection import train_test_split

model=LogisticRegression()
model.fit(x,y)
# sauvegarder le modéle entrainer avec joblib 
joblib.dump(model, 'data/churn_model_clean.pkl')
            
print('fin')
