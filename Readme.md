# Cyberbullying Detection in Tweets Using Machine Learning

## Project Overview  
This project focuses on detecting and classifying cyberbullying in tweets using machine learning models, aimed at addressing the growing issue of online harassment.

---

## Problem Statement  
1. Cyberbullying on social media platforms affects individuals' mental health and well-being.  
2. Tweets often contain noisy, informal text, making it difficult to detect and classify cyberbullying automatically.  
3. Current detection systems lack the ability to handle linguistic challenges like leetspeak and abbreviations.

---

## Objectives  
1. To preprocess tweet data effectively by addressing informal language issues.  
2. To train machine learning models to classify cyberbullying into specific categories.  
3. To deploy a real-time prediction system for detecting cyberbullying.  

---

## Dataset and Preprocessing  
- **Dataset:** A collection of tweets labeled with different types of cyberbullying.  
- **Preprocessing Steps:**  
  - Normalize leetspeak (e.g., replacing "h3ll0" with "hello").  
  - Remove URLs, mentions, and special characters.  
  - Tokenize, lowercase, and lemmatize the text to extract meaningful features.  
  - Use TF-IDF vectorization to convert text into numerical data for machine learning models.

---

## Machine Learning Models  
1. **Naive Bayes:** A probabilistic classifier effective for text-based tasks.  
2. **Logistic Regression:** A linear model suitable for binary and multi-class classification.  
3. **Decision Tree:** A non-linear model that creates a tree structure for decision-making.  
4. **Random Forest:** An ensemble model combining multiple decision trees for better accuracy.  

---

## Evaluation and Results  
- Each model was trained and evaluated using metrics like accuracy, confusion matrices, and classification reports.  
- The Random Forest model achieved the best accuracy, making it the top choice for deployment.

---

## Deployment via Flask  
1. **Backend:** Flask is used to create a web application that serves the trained models.  
2. **API Endpoint:** Accepts tweet text as input via a POST request and returns predictions from all models.  
3. **Preprocessing:** Ensures that incoming text is processed identically to the training data.  

---

## Methodology  
1. Load and preprocess data.  
2. Train multiple machine learning models using TF-IDF features.  
3. Save the trained models and vectorizer using `joblib`.  
4. Build a Flask app for real-time predictions, integrating the saved models.

---

## Impact  
1. Provides an automated system to identify cyberbullying, helping reduce online harassment.  
2. Enables organizations or individuals to monitor social media for harmful content.  
3. Offers a scalable, efficient solution that can adapt to evolving linguistic patterns in tweets.

---

## Future Improvements  
1. Integrate deep learning models like transformers for improved accuracy.  
2. Expand the dataset to include other languages and platforms.  
3. Add visualization tools to display trends in cyberbullying.
