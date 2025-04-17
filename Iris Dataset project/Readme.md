# 🌸 Iris Flower Classification Web App

This project is a simple and interactive **Streamlit web application** that predicts the species of an Iris flower based on four input features:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

The prediction is made using a **pre-trained machine learning model** and a **feature scaler**, both saved as `.pkl` files.

---

## 🔍 About the Dataset

The **Iris dataset** is one of the most well-known datasets in machine learning and contains 150 samples of Iris flowers across 3 species:

- **Setosa**
- **Versicolor**
- **Virginica**

Each sample has 4 numeric features:  
1. Sepal length (cm)  
2. Sepal width (cm)  
3. Petal length (cm)  
4. Petal width (cm)

---

## 🚀 Features

- Interactive sliders to input flower measurements
- Predicts the Iris species using a trained model
- Real-time results
- Lightweight and easy to deploy

---

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/NehaGhadge07/Machine-Learning/tree/main/Iris%20Dataset%20project
   cd iris-streamlit-app
   
Install dependencies:

pip install -r requirements.txt
Make sure you have the following files in the same directory:

model.pkl – Trained classification model

scaler.pkl – Scaler used to normalize the data

iris_predict_app.py – Streamlit app code

▶️ Run the App

streamlit run iris_predict_app.py
Then open the provided local URL in your browser.

📦 Dependencies
streamlit
numpy
scikit-learn
pickle (standard library)

📁 File Structure

iris-streamlit-app/
│
├── iris_predict_app.py     # Streamlit app code
├── model.pkl               # Trained ML model
├── scaler.pkl              # Scaler used for preprocessing
├── README.md               # This file
└── requirements.txt        # Python dependencies

🧠 Model Training (Optional)
The model and scaler were trained on the Iris dataset using RandomForestClassifier and StandardScaler from scikit-learn.

You can retrain the model if needed.

💡 Example Input

Sepal Length	Sepal Width	Petal Length	Petal Width
5.1	3.5	1.4	0.2
Predicted class: setosa

📌 License
This project is open-source and available under the MIT License.

🙋‍♀️ Author
Neha Ghadge
Connect with me on LinkedIn










