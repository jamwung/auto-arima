# auto-arima: Automated Time Series Forecasting with ARIMA Models

![Python](https://img.shields.io/badge/Python-3.7%20%7C%203.8%20%7C%203.9-blue)

Welcome to the `auto-arima` repository! This project is a personal attempt to create an automated ARIMA (AutoRegressive Integrated Moving Average) modeling pipeline for time series forecasting. It was developed as part of my Machine Learning 2 course.

## Overview

Time series forecasting is a critical aspect of data analysis, and ARIMA models are commonly used for this purpose. This repository incorporates various data preprocessing steps, including stationarity tests, differencing, and log transformations, to prepare time series data for modeling. Additionally, it provides visualizations of the Autocorrelation Function (ACF) for exploratory data analysis (EDA). The main focus is on automating the process of hyperparameter tuning and cross-validation of ARIMA models.

## Features

- **Data Preprocessing:** Auto-ARIMA automates the preparation of time series data by addressing issues related to stationarity and transformations.

- **Exploratory Data Analysis:** Visualizations of ACF help users gain insights into the autocorrelation structure of the time series.

- **Automated Modeling:** The project implements an automated machine learning pipeline for hyperparameter tuning and cross-validation of ARIMA models, making it easier to find the best-fitting model for your time series data.

## Getting Started

To use Auto-ARIMA for your time series forecasting:

1. Clone this repository to your local machine.

2. Ensure you have Python 3.7+ installed.

3. Install the required libraries using pip:

   ```shell
   pip install numpy pandas matplotlib statsmodels scikit-learn
