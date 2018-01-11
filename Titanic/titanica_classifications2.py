import pandas as pd

#COLETA DADOSS
train = pd.read_csv('train.csv')
teste = pd.read_csv('test.csv')

# REMOVE DADOS DESNECESSÁRIOS

train.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
teste.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# TRATA DADOS

new_data_train = pd.get_dummies(train)
new_data_test = pd.get_dummies(teste)

# Removendo valores nulos
new_data_train['Age'].fillna(new_data_train['Age'].mean(), inplace=True)
new_data_test['Age'].fillna(new_data_test['Age'].mean(), inplace=True)
new_data_test['Fare'].fillna(new_data_test['Fare'].mean(), inplace=True)



x = new_data_train.drop('Survived', axis = 1)
y = new_data_train['Survived']
#ML

from sklearn.ensemble import AdaBoostClassifier
modelo = AdaBoostClassifier()
modelo.fit(x, y)
print(modelo.score(x, y))

sub = pd.DataFrame()
sub['PassengerId'] = new_data_test['PassengerId']
sub['Survived'] = modelo.predict(new_data_test)

sub.to_csv('subsmission.csv', index=False)