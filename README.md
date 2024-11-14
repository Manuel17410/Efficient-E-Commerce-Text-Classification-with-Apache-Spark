# Efficient E-Commerce Text Classification with Apache Spark

This project focuses on text classification for an e-commerce dataset, categorizing products into four main classes: "Electronics," "Household," "Books," and "Clothing & Accessories." These categories cover around 80% of typical e-commerce offerings. The dataset, in CSV format, includes 50,425 entries with two columns: class name and product description. Using Apache Spark, a text classification algorithm was developed to efficiently classify the products into these categories, leveraging the dataset’s multivariate characteristics and real attribute values.

## Resources Used

**Python Version**: 3.12.1

**Packages**: pyspark, matplotlib, plotly

**Dataset**: https://www.kaggle.com/datasets/saurabhshahane/ecommerce-text-classification

## Data Cleaning

* Columns needed to be renamed, because Spark datasets asign automatically names to the columns that are not descriptive
* Null values were eliminated
* Some rows were eliminated, because the text information was transposing to the class column.

  ![Example Image](images/newplot(1).png)

## Feature Engineering

A pipeline was used to automate a series of transformation on the data.
It contained the following steps:

* StringIndexes: takes the column Class and converts the categories to numeric, so at the end, it is going to have values ranging from 0 to 3.
* Tokenizer: takes the column Comment, and transform the data in this way, for example, lets say that we have "I went running to Atlanta", after this transformation I will have [i, went, running, to, atlanta].
* StopWordsRemover: gets rid of the most common words, in this case, from the english language. Taken the example from before, the result would be [went, running, atlanta
* CountVectorizer: counts how many times a word appear in a document. For example, I have "i, am, manuel", and "i, am ,argentinian", this would be represented as (1,2,3,4). Lets say I have "I I Manuel", this would be (2,0,1,0) ---- Note: words are between , and are lowercase because it was previously tokenized.
* IDF: measures how important a word is in a document relative to a collection of documents. It combines two metrics:TF (Term Frequency) that counts how often a word appears in a document and IDF (Inverse Document Frequency) which reduces the weight of common words across many documents.

## Model Building

The chosen model was a Naive Bayes Classifier, which assumes that the features are conditionally independent of each other, given the class label—a "naive" assumption, since in real-world data, features often interact. Despite this, Naive Bayes often performs well in practice.

## Model Evaluation and Conclussion

The model performs exceptionally well, even on imbalanced data, achieving an overall accuracy of 0.964. Class-specific accuracies are also consistent, ranging from 0.948 to 0.986.
