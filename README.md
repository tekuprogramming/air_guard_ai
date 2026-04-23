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

This project uses public APIs:

* OpenWeather API (weather data)
* Open-Meteo Air Quality API (PM2.5, PM10)

No API key is required for air quality data.

Edit the `config.json` file:

```json
{
  "openweather_api_key": "YOUR_KEY"
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

## Data Sources and Data Collection

### Data Origin

The dataset used for training the air quality prediction model is collected in real time using external public APIs:

* **Weather data** is obtained from the **OpenWeatherMap API**
* **Air quality data (PM2.5, PM10) is obtained from the Open-Meteo Air Quality API**

### Data Collection Process

Data is collected using the script:

```
data_collection/collector.py
```

This script periodically retrieves:

* Temperature
* Humidity
* Pressure
* Wind speed and direction
* Cloud coverage
* AQI (Air Quality Index)

Each record is timestamped and stored in a CSV file:

```
historical_data.csv
```

### Data Structure

Each row in the dataset represents one measurement at a specific time and contains:

* Time features:

  * `timestamp`
  * `date`
  * `hour`
  * `day_of_week`
  * `is_weekend`

* Weather features:

  * `temperature`
  * `humidity`
  * `pressure`
  * `wind_speed`
  * `wind_deg`
  * `weather_main`
  * `clouds`

* Air quality:

  * `pm25`
  * `pm10`
  * `aqi_category (derived from PM2.5)`

### Data Preprocessing

Before training, the data undergoes several preprocessing steps:

* Removal of missing values (especially for PM2.5 and PM10 if available)
* Filtering unrealistic values (e.g. temperature range, pollutant limits)
* Interpolation of missing numeric values
* Encoding categorical variables (`weather_main`)
* Creation of time-based features (sin/cos transformations)
* Creation of target variable (`aqi_category_next`)

### Feature Scaling

Numerical features are standardized using StandardScaler:

- Mean = 0
- Standard deviation = 1

This improves model performance and ensures consistent feature ranges.

### Training and Test Data

* The dataset is split into:

  * **Training set (80%)**
  * **Test set (20%)**
* Stratified sampling is used to preserve class distribution
* The target variable is the **air quality category for the next time step**

---

## Data Processing Pipeline

1. Data collection from APIs
2. Cleaning and filtering invalid values
3. Handling missing values (linear interpolation)
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

### Target Variable

The model predicts the air quality category for the **next hour**.

This is created by shifting the original AQI category:

- Current data -> predicts future state

---

## How Training Data Was Created

The training dataset is not static. It is gradually built by continuously collecting real-time data from APIs.

Each new measurement is appended to the dataset, creating a time-series dataset used for model training.

The dataset grows over time, which improves model performance as more data becomes available.

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

---

## Model Training Notebook

https://colab.research.google.com/drive/1cRZ_h51JgoN0etRr9SV81PaHnHkJs26B#scrollTo=9RYlECzKVZFO
