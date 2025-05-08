# Unveiling Dhaka Housing Insights through AI-Powered Data Visualization

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Project Overview

This project explores the dynamics of the Dhaka housing market using a blend of data science techniques and advanced AI-powered visualizations. The primary objective is to uncover key patterns influencing property prices, segment the market, and understand feature interactions.  The analysis is presented through a modern, dark-themed visual narrative.

*Key Technologies:* Python, Pandas, Matplotlib, Seaborn, Plotly, Folium, Scikit-learn, SHAP

## Repository Structure

**Notebook**: The Jupyter Notebook Link (Open in New Tab) ([Unveiling_Dhaka_Housing_Insights_through_AI_Powered_Data_Visualization.ipynb](Unveiling_Dhaka_Housing_Insights_through_AI_Powered_Data_Visualization.ipynb)) --- includes the code and analysis.
* *README.md*: This file provides an overview of the project.

## Project Objectives

The project aims to:

1.  Perform comprehensive exploratory data analysis (EDA) to understand the distribution of features and their relationships with property prices.
2.  Preprocess the data to handle categorical and numerical features for machine learning.
3.  Develop a predictive model using RandomForestRegressor to estimate property prices.
4.  Apply AI-powered visualizations to:
   * Analyze geospatial patterns of property prices.
   * Segment the market using clustering techniques.
   * Explain the model's predictions using SHAP values.
   * Detect potential data anomalies.

## Dataset

A synthetic dataset of Dhaka housing listings is used, containing the following features:

* Latitude, Longitude: Geospatial coordinates.
* Area_sq_ft: Property size in square feet.
* Num_Bedrooms, Num_Bathrooms: Number of bedrooms and bathrooms.
* Age_of_Property_years: Age of the property.
* Floor_Level: Floor level of the property.
* Property_Type: Type of property (e.g., Apartment, Duplex).
* Location_Name: Name of the location in Dhaka (e.g., Gulshan, Mirpur).
* Lift_Available: Whether a lift is available.
* Parking_Space: Whether parking is available.
* Proximity_to_main_road_km: Proximity to main road in KM.
* nearby_amenities_score: Score indicating neaby ameneties.
* Price_BDT_Lakhs: Target variable - price in Lakhs Bangladeshi Taka.

## Key Findings

The project revealed several key insights into the Dhaka housing market:

* *Exploratory Data Analysis (EDA):*
   * Area_sq_ft, Num_Bedrooms, and Num_Bathrooms are positively correlated with Price_BDT_Lakhs.
   * Location_Name is a significant price driver, with areas like Gulshan and Baridhara commanding higher prices.
* *Property Segmentation:*
   * K-Means clustering identified distinct market segments, potentially representing different property types and price ranges (e.g., "Luxury Apartments in Prime Locations", "Budget-Friendly Housing").
* *Price Prediction:*
   * The RandomForestRegressor model accurately predicts property prices (Test R²: 0.9220, Test Adjusted R²: 0.9170, Test MAE: 74.81 Lakh BDT, Test RMSE: 101.77 Lakh BDT).
* *Model Explainability (SHAP):*
   * SHAP analysis confirmed that Area_sq_ft and Location_Name are the most important features in determining property prices.  Properties in Gulshan and Baridhara have a significant price premium.
* *Anomaly Detection:*
   * Isolation Forest identified potential pricing anomalies, which may warrant further investigation.

## Results

1.  *Exploratory Data Analysis (EDA)*
   * Initial exploration revealed significant drivers of housing prices.  Area_sq_ft, Num_Bedrooms, and Num_Bathrooms showed strong positive correlations with Price_BDT_Lakhs, while Age_of_Property_years had a slight negative correlation.
   * Crucially, Location_Name proved highly influential, with prime locations like Gulshan and Baridhara commanding substantially higher prices compared to areas like Mirpur or Old Dhaka.  Property types like 'Penthouse' and 'Duplex' also showed higher average prices.

2.  *Property Segmentation (K-Means)*
   * Using K-Means clustering on key property characteristics (Area, Bedrooms, Bathrooms, Age, Floor Level, Proximity, Amenities), the properties were successfully grouped into 4 distinct segments.
   * Cluster analysis highlighted differing profiles, potentially representing segments like 'Compact & Older', 'Spacious & Amenity-Rich', 'Mid-Range Balanced', etc., offering valuable insights for targeted marketing or market analysis.

3.  *Anomaly Detection (Isolation Forest)*
   * The Isolation Forest algorithm identified potential anomalies based on their feature combinations and pricing relative to the general patterns.

4.  *Price Prediction (Random Forest)*
   * A Random Forest Regressor model was developed to predict Price_BDT_Lakhs. Feature selection based on feature importance was implemented, resulting in a more parsimonious model. The final model demonstrated strong predictive performance.

   * --- Model Performance Metrics (RandomForestRegressor with Feature Selection) ---
   * Out-of-Bag (OOB) Score: 0.9792
   * Training R²: 0.9951
   * Test R²: 0.9220 (Explaining ~92.2% of the variance in test set prices).
   * Test Adjusted R²: 0.9170 (Confirming the relevance of selected features).
   * Test MAE: 74.81 Lakh BDT (Average absolute error in predictions).
   * Test RMSE: 101.77 Lakh BDT (Root mean squared error, sensitive to larger errors).

5.  *Model Explainability (SHAP)*
   * SHAP values provided crucial transparency into the "black box" Random Forest model.
   * Key Price Drivers: Confirmed Area_sq_ft as the most significant factor, followed by specific locations (e.g., Location_Name_Gulshan, Location_Name_Baridhara), Num_Bedrooms, and Nearby_Amenities_Score.

## Installation

1.  Clone the repository:

   
   git clone [https://github.com/zubyr09/Unveiling-Dhaka-Housing-Insights-through-AI-Visualization.git](https://github.com/zubyr09/Unveiling-Dhaka-Housing-Insights-through-AI-Visualization.git)
   

2.  Navigate to the project directory:

   
   cd Unveiling-Dhaka-Housing-Insights-through-AI-Visualization
   

3.  It is recommended to create a virtual environment:
   
    python -m venv venv
   
4.  Activate the virtual environment:

   * On Windows:
       
       venv\Scripts\activate
       
   * On macOS and Linux:
       
       source venv/bin/activate
       
5.  Install the required dependencies:

   
   pip install -r requirements.txt
   

## Usage

To reproduce the analysis and visualizations:

1.  Ensure you have followed the installation steps.
2.  Open the Jupyter Notebook:

   
   jupyter notebook notebooks/Unveiling_Dhaka_Housing_Insights_through_AI_Powered_Data_Visualization.ipynb
   

3.  Run the cells in the notebook sequentially.
4.  

## Author

zubyr09 - **Afridi Jubair**

## Acknowledgements

* This project was inspired by the need for better understanding of urban housing markets.
* The synthetic dataset was generated for demonstration purposes.
