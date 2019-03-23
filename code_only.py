import pandas as pd
from sklearn.model_selection import train_test_split
import re
from collections import Counter
import math

df = pd.read_csv('winemag-data_first150k.csv')
df = df[['description', 'points']]
df = df.drop_duplicates()
df = df.sort_values('points')
df['sentiment'] = df['points'].apply(lambda points: 0 if points < 88 else 1)
df = pd.concat([df[:10000], df[-10000:]])
df = df.drop('points', axis=1)
df_train, df_test = train_test_split(df, test_size=0.1)
df_train = df_train.copy()
df_test = df_test.copy()

def tokenize(description):
    return [token for token in re.split('\W+', description.lower()) if token]

df_train['tokens'] = df_train['description'].apply(tokenize)

lexicon = set([token for review in df_train['tokens'] for token in review])
negative = df_train[df_train['sentiment']==0]
positive = df_train[df_train['sentiment']==1]
negative_counts = Counter(token for review in negative['tokens'] for token in review)
positive_counts = Counter(token for review in positive['tokens'] for token in review)
positivity = {token: math.log((positive_counts[token]+1)/(negative_counts[token]+1)) for token in lexicon}

df_tokens = pd.DataFrame({'token': list(lexicon)})
df_tokens['positive_count'] = df_tokens['token'].apply(lambda token: positive_counts[token])
df_tokens['negative_count'] = df_tokens['token'].apply(lambda token: negative_counts[token])
df_tokens['positivity'] = df_tokens['token'].apply(lambda token: positivity[token])

strength = 2.8
df_strong = df_tokens[abs(df_tokens['positivity']) > strength]
def predict(tokens):
    score = df_strong[df_strong['token'].isin(tokens)]['positivity'].mean()
    return 1 if score > 0 else 0

df_train['prediction'] = df_train['tokens'].apply(predict)
correct = df_train[df_train['sentiment'] == df_train['prediction']]
train_accuracy = len(correct)/len(df_train)

df_test['tokens'] = df_test['description'].apply(tokenize)
df_test['prediction'] = df_test['tokens'].apply(predict)
correct = df_test[df_test['sentiment'] == df_test['prediction']]
test_accuracy = len(correct)/len(df_test)

print('Train accuracy: {}'.format(train_accuracy))
print('Test accuracy: {}'.format(test_accuracy))
