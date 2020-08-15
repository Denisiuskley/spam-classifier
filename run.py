from application import app
from spam_classifier import spam_classif
import pickle
import  pandas as pd

df = pd.read_csv('spam_or_not_spam.csv')
train_data = []
for i in range(df.shape[0]):
    train_data.append([df.iloc[i, 0], df.iloc[i, 1]])
    
a = spam_classif()
a.train(train_data)

with open('data.pickle', 'wb') as f:
    pickle.dump(a, f)
