##  Diabetes Prediction using Machine Learning

In this project, I developed a machine learning model to predict the likelihood of diabetes based on patient health data. I began by importing and exploring the Diabetes dataset, which contains key health indicators such as glucose level, blood pressure, insulin levels, BMI, and more.

The first step involved data preprocessing, where I cleaned the dataset, separated features from labels, and standardized the inputs using StandardScaler to ensure optimal model performance. I then split the data into training and test sets using train_test_split with stratified sampling for balanced class distribution.

For classification, I implemented a Support Vector Machine (SVM) with a linear kernel using Scikit-learn. The model was trained on the processed data, and its performance was evaluated using accuracy scores. The model achieved approximately 85.4% accuracy on the training set and 78.2% on the test set, indicating solid generalization.

Finally, I built a small predictive system that allows users to input custom values and receive a real-time prediction on whether the person is diabetic or not.

This project showcases my understanding of the end-to-end ML pipeline — from data preparation and model building to evaluation and deployment — using tools like Python, NumPy, Pandas, and Scikit-learn.

---

### 📊 Dataset

* Source: PIMA Indians Diabetes Dataset
* Features: Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, DPF, Age
* Label: `Outcome` (1 = Diabetic, 0 = Non-Diabetic)

---

### 🔧 Tech Stack

* Python (NumPy, Pandas)
* Scikit-learn (SVM, StandardScaler, Train-Test Split, Accuracy Score)

---

### 🚀 Workflow

1. **Data Preprocessing**

   * Loaded dataset with Pandas
   * Standardized features using `StandardScaler`

2. **Model Training**

   * Trained an SVM Classifier (linear kernel)
   * Achieved \~85.4% training and \~78.2% test accuracy

3. **Prediction System**

   * Built a system to predict diabetes from user input

---

### 🧪 Sample Prediction

```python
input_data = (5,166,72,19,175,25.8,0.587,51)
# Output: The person is diabetic
```

---

### ✅ How to Run

```bash
pip install numpy pandas scikit-learn
python diabetes_prediction.py
```
