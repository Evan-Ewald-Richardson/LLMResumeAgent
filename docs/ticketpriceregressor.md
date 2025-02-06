# Fair Ticket Price Regressor at The University of British Columbia

## Dates
- **End**: December 2024

## Links
- Project Report: [[Portfolio Link](https://www.evanrichardsonengineering.com/work/ticketpriceregression)]

## Summary
The **Fair Ticket Price Regressor** is a machine learning model designed to predict concert ticket prices by integrating event metadata, artist popularity, venue characteristics, and city demographics. The goal was to create a data pipeline capable of handling real-world, messy datasets from multiple sources, rather than relying on pre-cleaned data. The model was trained on **9,351 events** using data collected from the **Ticketmaster API, Spotify API, and the Simplemaps World Cities Dataset**. It utilized a **Random Forest Regressor**, achieving a **Mean Absolute Error (MAE) of 8.52** on the test set.

A key innovation in the project was the use of **BERT embeddings** to capture semantic relationships within event and venue text data, allowing for richer feature extraction in a structured machine learning pipeline.

## Technologies Used
- **Python (Pandas, Scikit-Learn, XGBoost, Random Forest)**
- **API Integration (Ticketmaster API, Spotify API)**
- **Natural Language Processing (BERT Embeddings, HDBSCAN Clustering)**
- **Feature Engineering & Regression Modeling**

## Details
This project was motivated by the **lack of transparency and consistency in event ticket pricing**, particularly for live concerts. By analyzing public ticket data and integrating multiple sources, the model aimed to offer both consumers and venues a **data-driven approach** to understanding ticket price fairness.

### **Evan's Contributions**
Evan was responsible for **the full data pipeline, model training, and feature engineering**, with a focus on integrating **text embeddings, clustering, and regression modeling**.

- **Data Collection & Processing**
  - Designed a **multi-source API scraping pipeline** that collected ticket prices, event details, and artist metadata.
  - **Cleaned and normalized data** across **Ticketmaster, Spotify, and city demographic datasets**.
  - Applied **outlier removal and feature transformations** to handle skewed pricing distributions.

- **Feature Engineering**
  - Applied **sine transformation** on temporal data to capture periodic trends in ticket pricing.
  - Implemented **mean encoding** of music genres for categorical feature representation.
  - Used **BERT-based embeddings** to cluster and categorize event and venue names using **HDBSCAN**, improving feature extraction from textual data.
  
- **Machine Learning Model Development**
  - Experimented with **Linear Regression, Ridge Regression, Random Forest, XGBoost, and Support Vector Regression**.
  - **Hyperparameter tuning** was performed, optimizing **tree depth, regularization, and min_samples splits**.
  - The **Random Forest model** outperformed other regressors with an MAE of **8.52**, effectively balancing complexity and interpretability.
  
- **Evaluation & Model Refinement**
  - Addressed **highly skewed ticket price distributions** but found **log transformations and Poisson loss functions were ineffective**.
  - Implemented **SHAP analysis** to understand feature importance, confirming that **text embeddings from venue names and event titles were among the most predictive variables**.
  - Investigated alternative loss functions and target transformations but found inherent feature limitations constrained further accuracy gains.

### **Project Impact & Future Improvements**
The **Fair Ticket Price Regressor** successfully demonstrated the ability to **predict concert ticket prices based on multi-source data**, validating the impact of **text embeddings in structured price forecasting**. Despite challenges with price skewness, the model achieved **actionable accuracy** for consumers and ticket sellers.

Potential future improvements include:
- **Expanding text-based data sources** by integrating **Wikipedia venue descriptions and Spotify artist biographies** for richer embeddings.
- **Leveraging deep learning approaches** such as fine-tuning transformer models to enhance text-based price prediction.
- **Incorporating historical ticket price trends** to improve model robustness across different event types.

This project highlights the power of **modern NLP techniques in classical machine learning settings**, demonstrating how **BERT embeddings and clustering techniques can enrich structured regression models**.
