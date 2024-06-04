# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
data = {
    'email': [
        "Hi, I saw your profile on LinkedIn and would like to connect.",
        "Congratulations! You've won a $1000 gift card. Click here to claim.",
        "Reminder: Your appointment is scheduled for tomorrow at 10 AM.",
        "Get a loan approved in minutes with low interest rates. Apply now!",
        "Please find the attached report for last month's sales."
    ],
    'label': ['ham', 'spam', 'ham', 'spam', 'ham']
}
df = pd.DataFrame(data)

# Data preprocessing
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Split the dataset into training and testing data
X = df['email']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Text processing and feature extraction
count_vectorizer = CountVectorizer()
tfidf_transformer = TfidfTransformer()

# Transform training data into tf-idf vectors
X_train_counts = count_vectorizer.fit_transform(X_train)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# Transform testing data into tf-idf vectors
X_test_counts = count_vectorizer.transform(X_test)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

# Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

# Support Vector Machine model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_tfidf, y_train)

# Predict and evaluate Naive Bayes model
y_pred_nb = nb_model.predict(X_test_tfidf)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
precision_nb = precision_score(y_test, y_pred_nb, pos_label='spam')
recall_nb = recall_score(y_test, y_pred_nb, pos_label='spam')
f1_nb = f1_score(y_test, y_pred_nb, pos_label='spam')

# Predict and evaluate SVM model
y_pred_svm = svm_model.predict(X_test_tfidf)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm, pos_label='spam')
recall_svm = recall_score(y_test, y_pred_svm, pos_label='spam')
f1_svm = f1_score(y_test, y_pred_svm, pos_label='spam')

# Print evaluation metrics
print("Naive Bayes Model:")
print(f"Accuracy: {accuracy_nb}")
print(f"Precision: {precision_nb}")
print(f"Recall: {recall_nb}")
print(f"F1 Score: {f1_nb}")

print("\nSupport Vector Machine Model:")
print(f"Accuracy: {accuracy_svm}")
print(f"Precision: {precision_svm}")
print(f"Recall: {recall_svm}")
print(f"F1 Score: {f1_svm}")
