# train_model.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import joblib

# 1. Load your dataset
df = pd.read_csv("news.csv")


# 2. Clean any missing values
df.dropna(subset=['text', 'label'], inplace=True)

print(df['label'].value_counts())

# 3. Features and Labels
X = df['text']
y = df['label']

# 4. Vectorize the text
vectorizer = TfidfVectorizer(stop_words='english')
X_vec = vectorizer.fit_transform(X)

# 5. Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42, stratify=y)

# 6. Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# 7. Save the model and vectorizer
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("✅ Model and vectorizer saved successfully!")
