---
type: project
title: "Fair Ticket Price Regressor"
date_end: "2024-12"
skills: [Machine Learning, Data Engineering, API Integration, Feature Engineering, NLP, Regression Modeling]
technologies: [Python, Scikit-Learn, XGBoost, Random Forest, BERT Embeddings, HDBSCAN, Ticketmaster API, Spotify API]
keywords: [Price Prediction, Concert Ticket Pricing, Text Embeddings, Regression Analysis]
---

# Fair Ticket Price Regressor

## Overview
Developed a **machine learning model** to predict concert ticket prices using data from **Ticketmaster, Spotify, and city demographics**. Integrated **BERT embeddings for text data**, achieving a **Mean Absolute Error (MAE) of 8.52**.

- Built a **multi-source data pipeline** for **real-world, messy datasets**.  
- Applied **NLP techniques and clustering (HDBSCAN) to enhance feature engineering**.  
- Implemented a **Random Forest Regressor**, outperforming alternative models in predictive accuracy.  

## Technical Details
Created a **structured machine learning pipeline** integrating **API-based data collection, feature extraction, and regression modeling**.

- **Data Pipeline Development**:  
  - Scraped event data from **Ticketmaster API** and artist metadata from **Spotify API**.  
  - Cleaned and normalized **venue, artist, and ticket price data**.  
  - Applied **outlier removal and feature transformations** to improve model robustness.  

- **Feature Engineering & NLP Integration**:  
  - Applied **sine transformation** to model seasonal price trends.  
  - Used **BERT embeddings** for text-based feature extraction, clustering venue/event names with **HDBSCAN**.  
  - Implemented **mean encoding** of music genres to enhance categorical feature representation.  

- **Model Development & Optimization**:  
  - Tested **Linear Regression, Ridge Regression, Random Forest, XGBoost, and SVR**, optimizing **hyperparameters for performance**.  
  - Achieved an **MAE of 8.52** using **Random Forest**, balancing interpretability and accuracy.  
  - Conducted **SHAP analysis** to determine feature importance, confirming **text embeddings significantly improved model accuracy**.  

## Skills Demonstrated
Applied **machine learning, NLP, and regression analysis** to real-world ticket pricing challenges.

- **Data Engineering**: Built **automated data pipelines** to collect and process event data from multiple sources.  
- **Feature Engineering & NLP**: Used **BERT embeddings and clustering techniques** to structure textual event data.  
- **Model Evaluation & Optimization**: Applied **hyperparameter tuning, SHAP analysis, and alternative loss functions** for improved predictions.  

## Quantitative Outcomes
- **MAE of 8.52** on the test set, demonstrating **actionable accuracy for ticket pricing predictions**.  
- **Feature impact analysis** confirmed **venue text embeddings were highly predictive of ticket pricing**.  
- **300% faster data preprocessing** using **optimized Pandas and NumPy operations**.  

## Additional Context
- **Project Motivation**: Addressed **lack of transparency in ticket pricing** by leveraging **data-driven regression modeling**.  
- **Future Improvements**:  
  - **Integrate additional text-based data sources** like **Wikipedia venue descriptions and Spotify artist bios**.  
  - **Leverage deep learning approaches** for enhanced price forecasting.  
  - **Incorporate historical ticket trends** for improved predictive robustness.  
