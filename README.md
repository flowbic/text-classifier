# Text classification of Wikipedia articles
Machine learning system that classifies Wikipedia texts and compares ML-techniques accuracy.
The Bag-of-words approach is used.

## Compares tecniques
Texts are classified with two different algorithms.
The algorithms are Multinomial Naïve Bayes and Linear SVC (Support Vector Machines).

These two algorithms are executed two times.
The first run executes the two algorithms and uses word count.
The second run executes both algorithms again, bu Term Frequency-Inverse Document Frequency (TF-IDF) is used to convert word count to word frequency.

## Execution
Example of executing the Machine Learning algorithms
![Screenshot of terminal, after execution](/images.screenshot.png)