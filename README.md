## Project Name: Customer Feedback Analysis Dashboard

An interactive web-based application built with Streamlit to process customer feedback send via a csv file, cleand and process the reviews, classify sentiment, extract key themes via clustering, and visualize insights using intuitive charts and word clouds.

## Features

- Upload and analyze customer feedback from CSV files.
- Automated text preprocessing which includes Clean and normalize text (lowercasing, removing stopwords, punctuation, lemmatization)
- While training  my original dataset, I handled imbalanced distribution of positive, negative, and neutral reviews using RandomUnderSampler
- Sentiment classification (Positive / Neutral / Negative) using a pre-trained TF-IDF + Logistic Regression model pipeline. 
- Topic modeling using K-Means clustering on TF-IDF vectors.
- Visual insights: a) Sentiment Overview: Pie chart showing percentage breakdown of Positive/Neutral/Negative comments.
-                  b) Topic Insights: Bar chart listing top keywords for each theme/topic cluster.
                    c) Feedback Table: Interactive table with original comment, sentiment label, and assigned topic.
                   d)  Word Clouds: Two word clouds—one for the most frequent words in positive comments, one for negative.
               
## Project Architecture

LQDIGITALPROJECT
1. app.py                         # Main Streamlit dashboard application
2. requirements.txt               # Python dependencies
3. .gitignore                    # Ignored files (venv)

4. data/

    4.1 trc_results.txt           # Cluster names and top keywords for each cluster of training data

5. datasets/

    5.1 Womens Clothing E-Commerce Reviews.csv    # Original dataset from Kaggle for training model

    5.2 Customer Reviews 2.csv                    # CSV file for user upload in Streamlit app

6. models/

    6.1 data_processed.pkl        # Dataset with sentiment columns added, redundant columns dropped

    6.2 data_cleaned.pkl          # Dataset after cleaning and lemmatization

    6.3 model.pkl                 # Sentiment classification pipeline

7. modules/
  
     7.1 preprocessing.ipynb       # Notebook for addition of Sentiment columns for supervised learning, dropping redundant columns in original dataset
     
     7.2 data_processor.py         # Python functions for cleaning and normalizing text
     
     7.3 export_model.ipynb        # Handling imbalanced data, TF-IDF + Logistic Regression pipeline, training/testing, evaluation
     
     7.4 find_topics.py            # TF-IDF + KMeans topic extraction for user-uploaded CSV
    
     7.5 predict.py                # Sentiment prediction logic
     
     7.6 tf_idf.py                 # TF-IDF + KMeans topic extraction for original CSV and cluster keyword file creation
     
     7.7 trc.png                   # Scatterplot of clusters for original dataset

8. venv/                         # Virtual environment (excluded from repo)


⚙️ Setup Instructions
1. Clone the Repository
  git clone  https://github.com/Salomiairy11/LQDigitalSentimentAnalysis.git
  cd LQDigitalSentimentAnalysis
2. Set Up Virtual Environment
  python -m venv venv
  venv\Scripts\activate (For Windows)
3. Install Dependencies
  pip install -r requirements.txt
4. Run the Streamlit App
  streamlit run app.py
5. Input Format
Upload a .csv file with at least 20 rows and one column containing free-text customer feedback, The feedback column name should be Full_Review. Or use the CSV file provided.

Notes: 
Pretrained models are stored in /models/.
All preprocessing logic is abstracted in modular files under /modules/.

