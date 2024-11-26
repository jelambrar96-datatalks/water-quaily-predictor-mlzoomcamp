# Water Quality Predictor 🌊

A machine learning model to predict water quality based on chemical and physical parameters using Python, scikit-learn, and pandas.

## Overview

This project implements a water quality classification system that predicts whether water samples are potable based on various water quality metrics. 

## Problem Description: Water Quality Potability Prediction

Access to clean and safe drinking water is essential for human health and well-being. However, many regions around the world face challenges in ensuring water quality due to natural and anthropogenic factors, such as industrial pollution, agricultural runoff, and inadequate water treatment systems. Monitoring water quality is crucial to determine its potability and protect public health. Traditional methods of testing water quality often rely on laboratory analysis, which can be time-consuming, costly, and inaccessible to resource-limited areas.

###  The Challenge:

Determining the potability of water requires evaluating various physicochemical and biological parameters, such as pH, turbidity, hardness, total dissolved solids (TDS), and the presence of contaminants like nitrates or heavy metals. Identifying potability manually involves collecting samples and performing detailed lab analyses, which may delay responses to potential health risks.

## Proposed Solution:

A machine learning classification model can be developed to predict water potability. The model will be trained on historical water quality datasets, such as those containing measurements of key indicators (e.g., pH, turbidity, and total chlorides) and labels indicating whether the water is potable.

A predictive machine learning model offers a powerful alternative to conventional methods. By analyzing patterns in historical water quality data, a machine learning model can predict whether a given sample is potable or non-potable based on measurable features. This approach has the potential to:

1. **Increase Efficiency**: Provide near-instantaneous predictions about water potability, reducing reliance on laboratory testing.
2. **Improve Accessibility**: Enable real-time decision-making in remote or resource-limited regions using portable sensors combined with predictive algorithms.
3. **Support Preventive Action**: Identify potential water safety issues early, allowing for rapid intervention and resource allocation to treat contaminated sources.

### Steps to Implementation:

1. **Data Collection**: Use publicly available datasets, such as those from government agencies or research initiatives, to gather labeled data on water quality parameters.
2. **Feature Engineering**: Identify relevant features influencing water potability and preprocess the data to handle missing values, outliers, or imbalanced classes.
3. **Model Development**: Train various classification models (e.g., logistic regression, decision trees, random forests, or neural networks) and evaluate their performance on a validation set.
4. **Deployment**: Deploy the best-performing model as an API or integrate it into an IoT-based water monitoring system for real-time predictions.
5. **Validation**: Continuously monitor and validate the model's predictions against new data to ensure reliability and accuracy.

## Expected Outcomes:

- Faster and more cost-effective determination of water potability.
- Increased accessibility to water safety evaluations in underserved areas.
- Enhanced ability to predict and prevent health risks associated with contaminated water supplies.

Machine learning models for water quality potability prediction can significantly contribute to public health initiatives and help achieve sustainable development goals related to clean water and sanitation.


##  Reproducitibility

This repository contains a machine learning project that includes data analysis, model training, and API deployment using Flask and Docker.

### Prerequisites

- Python 3.13.0
- Docker and Docker Compose
- Git
- pipenv
- Jupyter Notebook

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/jelambrar96-datatalks/water-quaily-predictor-mlzoomcamp
cd water-quaily-predictor-mlzoomcamp
```

### 2. Set Up Virtual Environment

Navigate to the notebooks directory and set up the virtual environment using pipenv:

```bash
cd notebooks
pipenv install
pipenv shell
```

This will create a new virtual environment and install all required dependencies from the Pipfile.

### 3. Launch Jupyter Notebook

Once inside the virtual environment, start Jupyter Notebook:

```bash
jupyter notebook
```

The Jupyter interface will open in your default web browser.

### 4. Execute Exploratory Data Analysis

Open and run `eda.ipynb` in Jupyter:
- This notebook contains exploratory data analysis
- Make sure to run all cells in sequence
- Review the analysis results and visualizations
- Any data preprocessing steps will be documented here

### 5. Train the Model

You have two options for training the model:

#### Option A: Using Jupyter Notebook
- Open and run `training.ipynb`
- Execute all cells sequentially

#### Option B: Using Python Script
- Run the training script directly:
```bash
python training.py
```

Both methods will generate:
- `dv.pkl`: Dictionary Vectorizer file
- Multiple model files with different algorithms/parameters

### 6. Prepare Flask Application

Copy the necessary model files to the Flask application directory:

```bash
# Create model directory if it doesn't exist
mkdir -p flask/model

# Copy the Dictionary Vectorizer
cp notebooks/dv.pkl flask/model/

# Copy your chosen model (rename it to model.pkl)
cp notebooks/[chosen_model].pkl flask/model/model.pkl
```

### 7. Deploy with Docker

Navigate to the project root directory and start the Docker containers:

```bash
docker-compose up -d
```

This will:
- Build the Docker image
- Start the Flask application
- Expose the API on port 9696

Verify the container is running:
```bash
docker ps
```

### 8. Test the API

Run the unittest suite to verify the API functionality:

```bash
cd flask/test
python -m unittest test_flask_app.py -v
```

The tests will verify:
- API endpoint accessibility
- Correct model predictions
- Error handling
- Response formats

## API Endpoints

- POST `/predict`
  - Accepts JSON input
  - Returns model predictions
  - Example request:
    ```bash
    curl -X POST http://localhost:9696/predict \
         -H "Content-Type: application/json" \
         -d '{"ph": 7.875895135481787, "hardness": 226.28478781681216, "solids": 12710.249451611751, "chloramines": 7.303126583151656, "sulfate": 346.4032581373675, "conductivity": 445.3741474328599, "organic_carbon": 6.063461912529135, "trihalomethanes": 63.12804403102223, "turbidity": 4.238589203481521}'
    ```

## Troubleshooting

Common issues and solutions:

1. If Jupyter doesn't start:
   - Verify pipenv environment is activated
   - Check if jupyter is installed: `pip install jupyter`

2. If Docker container fails to start:
   - Verify Docker daemon is running
   - Check ports are not in use: `lsof -i :9696`
   - Review logs: `docker-compose logs`

3. If model files are not generating:
   - Check available disk space
   - Verify write permissions in directories
   - Review training logs for errors

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

