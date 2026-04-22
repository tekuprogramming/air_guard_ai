# Air Quality Prediction System (Prague)

A machine learning system that collects, analyzes, and predicts air quality in Prague using real-world weather and pollution data.

---

## Project Overview

This project:

- Collects real-time weather and air quality data from external APIs
- Stores historical data for analysis
- Cleans and preprocesses datasets
- Trains a machine learning model to predict air quality categories
- Visualizes current and historical air quality trends

The goal is to predict the **next-hour air quality category** based on environmental conditions.

---

## Technologies Used

- Python 3.10+
- Pandas, NumPy (data processing)
- Scikit-learn (machine learning)
- Matplotlib, Seaborn (visualization)
- Requests (API communication)
- Pickle (model storage)

---

## Installation

### 1. Clone the project

```bash
git clone <repo-url>
cd project
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn requests
```

---

## API Setup

This project requires API keys:

* OpenWeather API
* IQAir API

Edit the `config.json` file:

```json
{
  "openweather_api_key": "YOUR_KEY",
  "iqair_api_key": "YOUR_KEY"
}
```

---

## How to Run

---

### 1. Start data collection

Collects weather and air quality data continuously:

```bash
python data_collection/collector.py
```

Data is saved into:

```
historical_data.csv
```

---

### 2. Train the model

```bash
python module_training/air_module_training.py
```

This step:

* cleans the dataset
* generates features
* trains Random Forest model
* saves model to `trained_model.pkl`

---

### 3. Make predictions

---

### 4. Generate visualizations

```bash
python application/src/visualizer.py
```

Creates:

* current air quality charts
* 24h trends
* weekly statistics
* dashboard overview

Saved in:

```
module_training/visualizations
```

---

## Machine Learning Model

### Algorithm:

* Random Forest Classifier

### Task:

* Predict air quality category for the next hour

### Input features:

* Temperature
* Humidity
* Pressure
* Wind speed
* Cloud coverage
* Time-based features (hour, day)
* Weather type

### Output:

* Air quality category:

  * good
  * moderate
  * unhealthy_sensitive
  * unhealthy
  * hazardous

---

## Data Processing Pipeline

1. Data collection from APIs
2. Cleaning and filtering invalid values
3. Handling missing values
4. Feature engineering (time + weather features)
5. Label encoding
6. Model training
7. Evaluation (accuracy, confusion matrix)
8. Model saving


## Model Performance

The model is evaluated using:

* Accuracy score
* Classification report
* Confusion matrix
* Feature importance analysis

---

## Notes

* You must collect data before training the model
* API keys are required
* First run may take time due to data gathering
* Model improves with more historical data

---

## Author

Project created for learning purposes:

* Data science
* Machine learning pipeline
* Real-world API integration
* Data visualization

---

## Future Improvements

* Deep learning model (LSTM for time series)
* Web dashboard (Flask / Streamlit)
* Live map visualization
* Deployment to cloud
* Real-time alerts system
