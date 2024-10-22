# Sentiment Analysis using PyTorch MLP
# ------------------------------------------------------

# Import necessary libraries
import pandas as pd
import numpy as np
import re
import nltk
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.corpus import stopwords
from collections import Counter

# Ensure reproducibility
import random
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Download NLTK data (if not already downloaded)
nltk.download('stopwords')

# ------------------------------------------------------
# Part 1: Data Loading and Preprocessing
# ------------------------------------------------------

# Step 1: Load the Dataset
try:
    amazon_df = pd.read_csv('amazon_cells_labelled.txt', delimiter='\t', header=None, names=['sentence', 'label'])
    imdb_df = pd.read_csv('imdb_labelled.txt', delimiter='\t', header=None, names=['sentence', 'label'])
    yelp_df = pd.read_csv('yelp_labelled.txt', delimiter='\t', header=None, names=['sentence', 'label'])
except FileNotFoundError:
    print("Dataset files not found. Please ensure the dataset files are in the current directory.")
    exit()

# Step 2: Data Analysis
df = pd.concat([amazon_df, imdb_df, yelp_df], ignore_index=True)

print("First 5 rows of the dataset:")
print(df.head())

print("\nMissing values in each column:")
print(df.isnull().sum())

print("\nNumber of duplicate rows:")
print(df.duplicated().sum())

print("\nLabel distribution:")
print(df['label'].value_counts())

# Step 3: Data Cleaning and Preprocessing
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove digits
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return words

df['tokens'] = df['sentence'].apply(preprocess_text)

# Step 4: Build Vocabulary and Vectorize Text
all_tokens = [token for tokens in df['tokens'] for token in tokens]
vocab_size = 1000
most_common_tokens = Counter(all_tokens).most_common(vocab_size)
word_to_idx = {word: idx for idx, (word, _) in enumerate(most_common_tokens)}

def vectorize_tokens(tokens, word_to_idx):
    vector = np.zeros(len(word_to_idx), dtype=np.float32)
    for token in tokens:
        if token in word_to_idx:
            vector[word_to_idx[token]] += 1
    return vector

df['vector'] = df['tokens'].apply(lambda x: vectorize_tokens(x, word_to_idx))

# Step 5: Split the Dataset Manually
X = np.stack(df['vector'].values)
y = df['label'].values

indices = np.arange(len(df))
np.random.shuffle(indices)

split_ratio = 0.8
split_index = int(len(df) * split_ratio)

train_indices = indices[:split_index]
test_indices = indices[split_index:]

X_train = X[train_indices]
X_test = X[test_indices]
y_train = y[train_indices]
y_test = y[test_indices]

# ------------------------------------------------------
# Part 2: Model Definition
# ------------------------------------------------------

class SentimentMLP(nn.Module):
    def __init__(self, input_size):
        super(SentimentMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

input_size = X_train.shape[1]
model = SentimentMLP(input_size)

# ------------------------------------------------------
# Part 3: Training Loop
# ------------------------------------------------------

X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).float().view(-1, 1)
X_test_tensor = torch.from_numpy(X_test).float()
y_test_tensor = torch.from_numpy(y_test).float().view(-1, 1)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 20
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# ------------------------------------------------------
# Part 4: Evaluation
# ------------------------------------------------------

model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    predicted = (outputs > 0.5).float()

y_test_np = y_test_tensor.numpy()
predicted_np = predicted.numpy()

def calculate_metrics(y_true, y_pred):
    correct = (y_true == y_pred).sum()
    accuracy = correct / len(y_true)
    return accuracy

accuracy = calculate_metrics(y_test_np, predicted_np)

print(f'\nEvaluation Metrics:')
print(f'Accuracy  : {accuracy * 100:.2f}%')
