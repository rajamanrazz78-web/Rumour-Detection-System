# Rumour Detection AI Tool

## Overview
The **Rumour Detection AI Tool** is a Natural Language Processing (NLP)-based project that analyzes social media posts, newspaper articles, and blog content to determine whether the information is a **rumour** (fake news) or **true**. This project integrates **Google's Fact Check API** for cross-verification, improving the credibility of predictions.

## Features
- **NLP-based classification**: Uses Machine Learning models to classify text as rumour or truth.
- **Google Fact Check API Integration**: Cross-verifies content with verified fact-checking sources.
- **User-Friendly Web Interface**: Allows users to input text and get real-time analysis.
- **Jupyter Notebook for Training**: Model training and testing are performed in Jupyter Notebook.
- **Flask-based Web App**: The model is deployed using a Flask server.
- **Interactive and Visually Appealing UI**: The front end is designed to be clean and user-friendly.

## Technologies Used
- **Python**
- **Jupyter Notebook** (for model development)
- **Natural Language Processing (NLP)**
- **Machine Learning (ML)** (Logistic Regression / Transformer-based Model)
- **Google Fact Check API** (for verification)
- **Flask** (for web deployment)
- **HTML, CSS, JavaScript** (for front-end development)

## Project Structure
```
RumourDetectionProject/
│-- model_training.ipynb   # Jupyter Notebook for training the ML model
│-- dataset/               # Contains dataset files used for training
│-- flask_app/
│   ├── static/            # CSS, JavaScript files for UI
│   ├── templates/         # HTML files for UI
│   ├── app.py             # Flask backend for serving requests
│   ├── model.pkl          # Trained machine learning model
│   ├── requirements.txt   # Python dependencies
│   ├── fact_check.py      # Google Fact Check API Integration
│-- README.md              # Project documentation
```

## Installation Guide
### Prerequisites
- Python 3.x
- Jupyter Notebook (for training)
- Flask (for running the web app)
- Google Fact Check API key (for verification)

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/RumourDetectionAI.git
cd RumourDetectionAI
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Train the Model (If needed)
If you want to retrain the model, open **model_training.ipynb** in Jupyter Notebook and run all the cells.
The trained model will be saved as **model.pkl**.

### Step 4: Run the Flask Web Application
```bash
cd flask_app
python app.py
```
Open **http://127.0.0.1:5000/** in your browser to access the web application.

## Sample Inputs for Testing
### True News Example:
```
"NASA confirms the successful landing of the Perseverance rover on Mars."
```
### Fake News Example:
```
"5G towers are causing COVID-19 infections."
```

## Future Improvements
- Implement a **Deep Learning-based Transformer Model** for better accuracy.
- Improve **dataset quality** by integrating real-time news feeds.
- Add **multi-language support** for global usage.
- Optimize **Google Fact Check API requests** to handle large-scale data.

## Contributing
If you would like to contribute, feel free to fork the repository and submit a pull request.

## License
This project is licensed under the MIT License.

## Contact
For any queries, feel free to contact:
- **Vinamrakumar Vishwakarma**
- **vinamravishwakarma2004@gmail.com**
- **Github Profile : vinamra001**

---
This **README.md** provides a clear understanding of the project and instructions for setup and usage. You can modify the sections according to your project needs! 🚀

