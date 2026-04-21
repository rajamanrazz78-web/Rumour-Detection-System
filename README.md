Rumour Detection System
A Flask-based web application that predicts whether a news statement or article is a rumor or true news using Natural Language Processing and a trained Logistic Regression model. The project also integrates the Google Fact Check Tools API to fetch related fact-check results for supported claims.

Features
News text classification using TF-IDF and Logistic Regression.
​

Flask web interface for user input and result display.

Fact-check lookup using the Google Fact Check Tools API.
​

Confidence score display for model predictions.

Handles short inputs with warning messages.

Uses NLTK-based text preprocessing.

Project Structure
text
Rumour-Detection-System-main/
├── app.py
├── rumour.ipynb
├── logistic_regression_model.pkl
├── tfidf_vectorizer.pkl
├── templates/
│   ├── index.html
│   └── result.html
├── static/
│   └── style.css
└── README.md
Tech Stack
Python

Flask

scikit-learn

NLTK

HTML/CSS

Google Fact Check Tools API
​

How It Works
The user enters a news statement or article in the web form.

The application cleans the text using regex, tokenization, and stopword removal.

The TF-IDF vectorizer transforms the cleaned text into numerical features.

The Logistic Regression model predicts whether the content is rumor or true news.

The Google Fact Check Tools API is queried with a short claim to find related fact-check results.
​

The final result is shown with model prediction, confidence score, and API-based fact-check output.

Installation
Clone the repository:

bash
git clone https://github.com/your-username/your-repo-name.git
cd Rumour-Detection-System-main
Create and activate a virtual environment:

bash
python -m venv .venv
.venv\Scripts\activate
Install dependencies:

bash
pip install flask scikit-learn nltk requests
Download required NLTK resources:

python
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
API Key Setup
For security, do not hardcode your real API key in app.py. Use an environment variable instead.

Update your code like this:

python
import os
FACT_CHECK_API_KEY = os.environ.get("FACT_CHECK_API_KEY", "your_api_key")
FACT_CHECK_API_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
Set the key in PowerShell before running the app:

powershell
$env:FACT_CHECK_API_KEY="your_real_key_here"
python app.py
Running the Project
Start the Flask server:

bash
python app.py
Then open the local URL shown in the terminal, usually:

text
http://127.0.0.1:5000/
Sample Test Inputs
Rumor examples
Bill Gates put microchips in COVID-19 vaccines.

The 2020 U.S. presidential election was stolen through widespread voter fraud.

Climate change is a hoax.

True news examples
Chandrayaan-3 successfully landed on the Moon.

Regular UPI transactions are free for users.

Joe Biden won the 2020 U.S. presidential election.

Git Commands
To update your project on GitHub:

bash
git status
git add app.py rumour.ipynb README.md
git commit -m "Update app, notebook, and README"
git push
Notes
Fact Check API works best with short, specific claims rather than long paragraphs.
​

If you get a scikit-learn pickle error such as LogisticRegression object has no attribute multi_class, retrain or resave the model in the same virtual environment used by Flask.
​

Avoid uploading real API keys, large datasets, or unnecessary generated images to GitHub.
​

Future Improvements
Add support for Indian and international datasets.

Improve UI and result visualization.

Deploy the project on Render or another cloud platform.

Replace pickle-based loading with a more robust deployment workflow.

License
This project is for educational and academic use.
