# MigrationDetection

# Migration Data Analysis

This project involves analyzing migration data, exploring trends, and building predictive models using machine learning techniques. The dataset includes information on arrivals, departures, and net migration across different countries and citizenships from 1979 onwards.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Data Preprocessing](#data-preprocessing)
4. [Model Building](#model-building)
5. [Model Evaluation](#model-evaluation)
6. [Visualization](#visualization)
7. [Reproducing the Results](#reproducing-the-results)

## Project Overview

The goal of this project is to analyze migration data, understand historical trends, and build predictive models to forecast migration values. We use a Random Forest Regressor to predict migration values based on features like country, measure (arrivals, departures, net), year, and citizenship.

## Dataset Description

The dataset contains the following columns:

- `Measure`: Type of migration measure (Arrivals, Departures, Net)
- `Country`: Country of migration
- `Citizenship`: Citizenship of migrants
- `Year`: Year of record
- `Value`: Migration value

## Data Preprocessing

1. **Label Encoding**: Convert categorical columns (`Measure`, `Country`, `Citizenship`) into numerical labels.
2. **Handling Missing Values**: Fill missing values in the `Value` column with the median value.
3. **Feature Engineering**: Create new features (`CountryID`, `CitID`) by factorizing the `Country` and `Citizenship` columns.

## Model Building

We use a Random Forest Regressor with the following parameters:
- `n_estimators`: 70
- `max_features`: 3
- `max_depth`: 5
- `n_jobs`: -1

### Training the Model

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

X = data[['CountryID', 'Measure', 'Year', 'CitID']].values
Y = data['Value'].values

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=9)

rf = RandomForestRegressor(n_estimators=70, max_features=3, max_depth=5, n_jobs=-1)
rf.fit(X_train, y_train)
```

## Model Evaluation

Evaluate the model using the R² score:

```python
score = rf.score(X_test, y_test)
print(f"R² Score: {score}")
```

The model achieved an R² score of approximately 0.74, indicating a good fit to the data.

## Visualization

### Migration Trends for Oceania (CountryID = 0)

```python
import matplotlib.pyplot as plt

country_0_data = data[data['CountryID'] == 0]

arrivals = country_0_data[country_0_data['Measure'] == 'Arrivals']
departures = country_0_data[country_0_data['Measure'] == 'Departures']

grouped_arrivals = arrivals.groupby(['Year']).aggregate({'Value': 'sum'})
grouped_departures = departures.groupby(['Year']).aggregate({'Value': 'sum'})

plt.plot(grouped_arrivals.index, grouped_arrivals['Value'], marker='o', label='Arrivals', color='blue')
plt.plot(grouped_departures.index, grouped_departures['Value'], marker='o', label='Departures', color='green')

plt.title('Migration Trends for Oceania (CountryID = 0)')
plt.xlabel('Year')
plt.ylabel('Total Migration Value')
plt.legend()
plt.show()
```

### Growth of Migration to the USA by Year

```python
grouped = data.groupby(['Year']).aggregate({'Value': 'sum'})
grouped.plot(kind='line')
plt.axhline(169, color='g')
plt.title('Growth of Migration to the USA by Year')
plt.xlabel('Year')
plt.ylabel('Total Migration Value')
plt.show()
```

## Reproducing the Results

To reproduce the results in this repository, follow these steps:

1. **Clone the Repository**

```bash
git clone https://github.com/MeghavathMahender/migration-data-analysis.git
cd migration-data-analysis
```

2. **Install Dependencies**

Ensure you have Python installed, then install the required packages:

```bash
pip install pandas scikit-learn matplotlib
```

3. **Run the Jupyter Notebook**

Open and run the provided Jupyter notebook to see the data preprocessing, model training, evaluation, and visualization steps.

---
