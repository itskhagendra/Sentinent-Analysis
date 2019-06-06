from modules import ingest, cleaning, tokenize, bow_features
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = ingest()

n = data.shape[0]
data = cleaning(data)
print(data.index)

data = tokenize(data)
print(data[['sentiment', 'sentimentText', 'tidy']])

"""
print(data.head())


bow = bow_features(data)
for i in bow.columns:
    print(i)


print(bow.head())
train_bow = bow[:31962, :]
test_bow = bow[31962:, :]
x_train_bow, x_valid_bow, y_train, y_valid = train_test_split(train_bow, bow['tidy'][bow['sentiment']],
                                                              random_state=42, test_size=0.3)
lreg = LogisticRegression()
lreg.fit(x_train_bow, y_train)

pred = lreg.predict(x_valid_bow)
acc = accuracy_score(y_valid, pred)
print(acc)


print(bow.index)
"""

