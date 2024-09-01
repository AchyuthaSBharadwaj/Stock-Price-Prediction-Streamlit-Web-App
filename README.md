# Stock-Price-Prediction-Streamlit-Web-App
The Stock Price Prediction Web App uses machine learning (Facebook Prophet and Linear Regression) to forecast Amazon's stock prices based on historical data. The project provides an interactive Streamlit interface for visualizing trends and comparing model performance

## Features

- **Comparative Analysis**: Compares the performance of Facebook Prophet and Linear Regression models.
- **Interactive UI**: Allows users to choose the number of years for future predictions.
- **Visualizations**: Displays both raw data and model predictions using interactive Plotly graphs.
- **Evaluation Metrics**: Provides evaluation metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R2) for both models.

## Installation

To run this project locally, you'll need to have Python installed. Then, follow these steps:

1. **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2. **Install the required Python libraries:**

    You can install the required dependencies using `pip`:

    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Streamlit app:**

    ```bash
    streamlit run your_script_name.py
    ```

    Replace `your_script_name.py` with the name of your Python script containing the Streamlit app (e.g., `app.py`).

## Usage

- Open the app in your browser, where you can see the Amazon stock data, visualize it, and choose how many years ahead you want to predict the stock prices.
- The app will display the results from both the Facebook Prophet and Linear Regression models.
- Evaluate the models using the provided metrics, and observe the forecasted trends on the interactive plots.

## Data

The app uses the Amazon stock price data provided in a CSV file (`amazon.csv`). Ensure this file is in the same directory as your Streamlit app script.
