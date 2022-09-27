# E-commerce-products-Text-Classification

General info
The goal of the project is product categorization based on their description with Machine Learning  algorithms. Additionaly we have created Doc2vec  models and EDA analysis (data exploration, data aggregation and cleaning data).

Problem Statement
Classify the description of E-commerce products into 4 categories by implementing any NLP approach for analysis and modeling on the provided dataset. The objective is to recognize whether the given description is related to Electronics (0), Households (1), Books (2), or Clothing & Accessories (3) products. Focus majorly on unique preprocessing techniques.

Dataset
The dataset comes from https://www.kaggle.com/datasets/saurabhshahane/ecommerce-text-classification .

Motivation
The aim of the project is multi-class text classification to E-commerce products based on their description. Based on given text as an input, we have predicted what would be the category. We have four types of categories corresponding to different E-commerce products. In our analysis we used a different methods for a feature extraction (such as Word2vec, Doc2vec) and various Machine Learning/Deep Lerning algorithms to get more accurate predictions and choose the most accurate one for our issue.

Project contains:
Text classification with Doc2vec model -Text_Classification_Using_Doc2vec.ipynb
EDA analysis - Products_analysis.ipynb
data, models - data and models used in the project.

Summary
We begin with data analysis and data pre-processing from our dataset. Then we have used a few combination of text representation such as DBoW  and we have trained the  doc2vec models from our data. We have experimented with several Machine Learning algorithms: Logistic Regression, Linear SVM using different combinations of text representations and embeddings. 

From our experiments we can see that the tested models give a overall high accuracy and similar results for our problem. The SVM (DBOW ) model  give the best accuracy of validation set. Logistic regression performed very well both with DBOW and Doc2vec. We achieved an accuracy on the test set equal to 95 %. That shows the extensive models are not gave a better results to our problem than simple Machine Learning models such as SVM.

Model	Embeddings	Accuracy
SVM	Doc2vec (DBOW)	0.95
SVM	Doc2vec (DM)	0.86
Logistic Regression	Doc2vec (DBOW)	0.95
Logistic Regression	Doc2vec (DM)	0.86
The project is created with:
Python 3.8/3.10
libraries: NLTK, gensim, Keras, TensorFlow, Hugging Face transformers, scikit-learn, pandas, numpy, seaborn, pyLDAvis.

