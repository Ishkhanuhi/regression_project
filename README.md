# Regression Project

## Data Gathering and Cleaning

The data was collected from the YouTube's playlist, the one that we used during out lectures.

Then the raw data is written in the file named `data.csv`.

Afterwards, we are going to clean the raw data.

For that purpose, we use `nltk` library to remove punctuations, stopwords, numbers, etc.
Cleaned data is saved in a file named `cleaned_data.csv`.

Now we need to calculate `tf-idf` for each word in the documents.

### **How is `TF-IDF` calculated ?**

![alt text](img.png)

`TF-IDF` for a word in a document is calculated by multiplying two different metrics:

- The **term frequency** of a word in a document. There are several ways of calculating this frequency, with the
  simplest
  being a raw count of instances a word appears in a document. Then, there are ways to adjust the frequency, by length
  of a document, or by the raw frequency of the most frequent word in a document.
- The **inverse document frequency** of the word across a set of documents. This means, how common or rare a word is in
  the
  entire document set. The closer it is to 0, the more common a word is. This metric can be calculated by taking the
  total number of documents, dividing it by the number of documents that contain a word, and calculating the logarithm.
- So, if the word is very common and appears in many documents, this number will approach 0. Otherwise, it will approach
  to 1.

Next, we are going to perform sentiment analysis.
I have used `afinn` library to assign sentiment values to each document from `cleaned_data.csv`.
The result from the analysis is saved into `data_with_sentiment_analysis.csv`.

## Data Analysis and Regression

We're going to analyse the relation between 
