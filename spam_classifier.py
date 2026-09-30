import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Load SMS Spam Collection dataset directly from URL
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'text'])

# Convert labels to binary: ham (legitimate) = 0, spam = 1
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# 2. Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label_num'], test_size=0.2, random_state=42
)

# 3. Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 4. Train Naive Bayes Classifier
clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)

# 5. Evaluate Model
y_pred = clf.predict(X_test_tfidf)
print(f"Spam Classifier Test Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))