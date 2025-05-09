# Spam Detection Baseline Model

This project implements a baseline text classification model to detect spam emails using various text preprocessing techniques and the Naive Bayes algorithm. The dataset used is the [Spam-Ham Dataset from Kaggle](https://www.kaggle.com/datasets/venky73/spam-mails-dataset).

## 📁 Dataset

The dataset contains labeled email messages classified as either `spam` or `ham`. It can be downloaded from [here](https://www.kaggle.com/datasets/venky73/spam-mails-dataset?select=spam_ham_dataset.csv).

## 🧪 Features & Techniques

This notebook demonstrates the following steps:

* Baseline model using **CountVectorizer** and **Multinomial Naive Bayes**
* Basic **text preprocessing** (lowercasing, removing punctuation/numbers, lemmatization)
* **Stopword removal**
* Using **TF-IDF vectorization** instead of BoW
* Model evaluation with **accuracy**, **precision**, **recall**, **F1 score**, and **classification report**

## 🔧 Installation

Before running the notebook, make sure to install the following dependencies:

```bash
pip install pandas scikit-learn matplotlib seaborn nltk
```

Also, download necessary NLTK resources:

```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
```

## 🚀 Running the Code

1. Download the dataset CSV and update the file path in the code (`/content/spam_ham_dataset.csv`).
2. Run the script or Jupyter Notebook cell by cell to see each technique applied and its performance.
3. Check console outputs for detailed evaluation metrics and feature comparisons.

## 📊 Output

Each model variation prints:

* Classification performance (Accuracy, Precision, Recall, F1)
* Classification report
* Feature comparison (e.g., before and after stopword removal or TF-IDF transformation)

## 🛠️ Models Used

* **Multinomial Naive Bayes** for baseline and enhanced models
* **CountVectorizer** and **TfidfVectorizer** for feature extraction

## 📈 Future Improvements

* Explore more ML models (e.g., SVM, Logistic Regression)
* Incorporate n-grams and custom tokenization
* Use deep learning models like LSTM or Transformers
