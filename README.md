# Hotel Review Sentiment Analyzer

This Python project leverages the Natural Language Toolkit (NLTK) to perform sentiment analysis on hotel reviews.

### Dataset

`hotel-reviews.csv` (included) has 499 hotel reviews. Each one is labeled positive (`1`) or not positive (`0`) in the `Positive` column. 330 are labeled positive and 169 are not. The file also records the reviewer ID, the browser used, and the device type.

### How it works

All of the work is in [`sa-hotel-reviews.ipynb`](sa-hotel-reviews.ipynb):

1. **Preprocess the text.** Lowercase each review, split it into words, remove English stop words, and lemmatize what's left (for example, `rooms` becomes `room`).
2. **Score the sentiment.** Run each review through NLTK's VADER sentiment analyzer. A review is predicted positive if VADER finds any positive wording in it.
3. **Compare with the labels.** Check the predictions against the `Positive` column using scikit-learn's classification report.

### Results

| | Precision | Recall | F1 | Reviews |
|---|---|---|---|---|
| Not positive (0) | 0.50 | 0.01 | 0.02 | 169 |
| Positive (1) | 0.66 | 0.99 | 0.80 | 330 |

Overall accuracy is **66%**.

Almost every review contains some positive wording, even critical ones such as "The room was kind of clean but had a VERY strong smell of dogs." As a result, the model labels nearly everything as positive and catches only 1% of the negative reviews. About 66% of the reviews are positive, so labeling every review positive would also score about 66%.

### Ideas for improvement

- Use VADER's `compound` score with a cutoff instead of treating any positive wording as a positive review.
- Score the original text. VADER is designed to read capital letters, punctuation, and negations like "not", which the preprocessing step removes.
- Train a classifier, such as logistic regression on TF-IDF features, using the labeled reviews.

### How to run

```bash
pip install pandas nltk scikit-learn jupyter
python -m nltk.downloader punkt stopwords wordnet vader_lexicon
jupyter notebook sa-hotel-reviews.ipynb
```

### Technologies

**[Python](https://www.python.org/)**

**[NLTK](https://www.nltk.org/)**

**[scikit-learn](https://scikit-learn.org/)**
