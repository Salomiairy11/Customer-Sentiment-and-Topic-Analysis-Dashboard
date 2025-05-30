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
app.py                     # Main Streamlit dashboard application
requirements.txt           # Python dependencies
.gitignore                 # Ignored files (venv)
│
├── data/
│   ├── trc_results.txt        #Cluster names and top keywords for each clusters of my training data
│
├── datasets/                  
├──    ├──Womens Clothing E-Commerce Reviews.csv    #original dataset taken from kaggle for training model 
├──     ├── Customer Reviews 2 .csv                 #CSV file for user upload in Streamlit app
│
├── models/
│   ├── data_processed.pkl       # Original dataset with addition of Sentiment columns with positive/negative/neutral values for supervised learning, dropping redundant columns
│   ├── data_cleaned.pkl         # Dataset after cleaning and lemmatization 
│   └── model.pkl                # sentiment classification pipeline
│
├── modules/                   
|   |   preprocessing.ipynb    # Notebook for addition of Sentiment columns with positive/negative/neutral values for supervised learning, dropping redundant columns in original dataset
│   ├── data_processor.py      # python functions for cleaning and normalizing text
|   ├── export_model.ipynb     # handling imbalanced dataset, forming TD-IDF and Logistic Regression pipeline, Training and testing data, evaluation of Metrics
│   ├── find_topics.py         # TF-IDF + KMeans topic extraction functions for user uploaded csv 
│   ├── predict.py             # Sentiment prediction logic
│   ├── tf_idf.py              # TF-IDF + KMeans topic extraction for original csv and formation of file consisting of cluster name and keywords 
│   └── trc.png                # Scatterplot of clusters of original dataset
│
└── venv/                      # Virtual environment (not included in repo)

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

