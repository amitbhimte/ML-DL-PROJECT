##  Diabetes Prediction using Machine Learning

A machine learning project using **Support Vector Machine (SVM)** to predict whether a person is diabetic based on medical parameters.

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
