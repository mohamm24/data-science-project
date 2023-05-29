import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv(r"D:\security\tmdb_movies_data.csv")

# Dropping irrelevant column
df = df.drop(['id','imdb_id','homepage','tagline','overview','release_date','original_title'],axis=1)

# Handling missing values
df = df.dropna()

# Categorical to numerical conversion
df['genres'] = df['genres'].apply(lambda x: x.split(',')[0])
df['production_companies'] = df['production_companies'].apply(lambda x: x.split(',')[0])

# Splitting the data into features and target variable
X = df.drop(['revenue'],axis=1)
y = df['revenue']

# One-hot encoding
X = pd.get_dummies(X)

# Convert revenue into binary labels
y = np.where(y > y.median(), 1, 0)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initializing the model
model = LogisticRegression()

# Training the model
model.fit(X_train, y_train)

# Evaluating the model
y_pred = model.predict(X_test)

print('Accuracy:', accuracy_score(y_test, y_pred))