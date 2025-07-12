# Heart Disease Prediction System 🫀

A comprehensive machine learning project for predicting heart disease using various medical attributes and patient data. This system employs multiple algorithms to provide accurate predictions that can assist healthcare professionals in early diagnosis and treatment planning.

## 🌟 Project Overview

Heart disease remains one of the leading causes of death globally. Early detection and prediction can significantly improve patient outcomes and reduce healthcare costs. This project leverages machine learning techniques to analyze patient data and predict the likelihood of heart disease based on various medical parameters.

## 📊 Dataset Information

The project utilizes a comprehensive heart disease dataset containing the following features:

### Input Features:
- **Age**: Age of the patient (years)
- **Sex**: Gender (1 = Male, 0 = Female)
- **Chest Pain Type (cp)**: 
  - 0: Typical angina
  - 1: Atypical angina
  - 2: Non-anginal pain
  - 3: Asymptomatic
- **Resting Blood Pressure (trestbps)**: Resting blood pressure in mm Hg
- **Cholesterol (chol)**: Serum cholesterol level in mg/dl
- **Fasting Blood Sugar (fbs)**: Fasting blood sugar > 120 mg/dl (1 = True, 0 = False)
- **Resting ECG (restecg)**: Resting electrocardiographic results
  - 0: Normal
  - 1: ST-T wave abnormality
  - 2: Left ventricular hypertrophy
- **Max Heart Rate (thalach)**: Maximum heart rate achieved during exercise
- **Exercise Induced Angina (exang)**: Exercise induced angina (1 = Yes, 0 = No)
- **ST Depression (oldpeak)**: ST depression induced by exercise relative to rest
- **Slope**: Slope of the peak exercise ST segment
- **Number of Major Vessels (ca)**: Number of major vessels colored by fluoroscopy (0-3)
- **Thalassemia (thal)**: Thalassemia type
  - 1: Normal
  - 2: Fixed defect
  - 3: Reversible defect

### Target Variable:
- **Heart Disease (target)**: Presence of heart disease (1 = Disease, 0 = No Disease)

## 🛠️ Technologies Used

- **Python 3.8+**
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib**: Data visualization
- **Seaborn**: Statistical data visualization
- **Scikit-learn**: Machine learning algorithms
- **Jupyter Notebook**: Interactive development environment

## 🔧 Installation & Setup

1. **Clone the repository:**
```bash
git clone https://github.com/dixituday31/Heart-Disease-Prediction.git
cd Heart-Disease-Prediction
```

2. **Create a virtual environment:**
```bash
python -m venv heart_disease_env
source heart_disease_env/bin/activate  # On Windows: heart_disease_env\Scripts\activate
```

3. **Install required packages:**
```bash
pip install -r requirements.txt
```

4. **Launch Jupyter Notebook:**
```bash
jupyter notebook
```

## 📁 Project Structure

```
Heart-Disease-Prediction/
│
├── data/
│   ├── heart_disease_dataset.csv
│   └── processed_data.csv
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── utils.py
│
├── models/
│   ├── logistic_regression_model.pkl
│   ├── random_forest_model.pkl
│   ├── svm_model.pkl
│   └── best_model.pkl
│
├── visualizations/
│   ├── correlation_heatmap.png
│   ├── feature_importance.png
│   └── model_comparison.png
│
├── requirements.txt
├── README.md
└── main.py
```

## 🚀 Usage

### Quick Start
```python
# Load and run the main prediction script
python main.py
```

### Step-by-Step Analysis
1. **Data Exploration**: Open `notebooks/01_data_exploration.ipynb`
2. **Data Preprocessing**: Run `notebooks/02_data_preprocessing.ipynb`
3. **Model Training**: Execute `notebooks/03_model_training.ipynb`
4. **Model Evaluation**: View results in `notebooks/04_model_evaluation.ipynb`

### Making Predictions
```python
import pickle
import numpy as np

# Load the trained model
with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Example patient data
patient_data = np.array([[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]])

# Make prediction
prediction = model.predict(patient_data)
probability = model.predict_proba(patient_data)

print(f"Heart Disease Prediction: {'Positive' if prediction[0] == 1 else 'Negative'}")
print(f"Confidence: {max(probability[0]) * 100:.2f}%")
```

## 🤖 Machine Learning Models

This project implements and compares multiple machine learning algorithms:

### 1. **Logistic Regression**
- Linear model for binary classification
- Provides probability estimates
- Interpretable coefficients

### 2. **Random Forest**
- Ensemble method using multiple decision trees
- Handles feature importance well
- Robust to overfitting

### 3. **Support Vector Machine (SVM)**
- Effective for high-dimensional data
- Uses kernel trick for non-linear patterns
- Good generalization performance

### 4. **K-Nearest Neighbors (KNN)**
- Instance-based learning algorithm
- Simple and effective for small datasets
- No assumptions about data distribution

### 5. **Naive Bayes**
- Probabilistic classifier
- Fast and efficient
- Works well with categorical features

### 6. **Decision Tree**
- Tree-like model for decision making
- Easy to interpret and visualize
- Handles both numerical and categorical data

## 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| Random Forest | 87.5% | 0.89 | 0.85 | 0.87 | 0.92 |
| Logistic Regression | 85.2% | 0.86 | 0.84 | 0.85 | 0.90 |
| SVM | 83.7% | 0.85 | 0.82 | 0.83 | 0.88 |
| KNN | 81.3% | 0.83 | 0.79 | 0.81 | 0.85 |
| Naive Bayes | 79.8% | 0.81 | 0.78 | 0.79 | 0.84 |
| Decision Tree | 78.5% | 0.80 | 0.77 | 0.78 | 0.82 |

## 🔍 Key Insights

### Feature Importance Analysis
1. **Chest Pain Type**: Most significant predictor
2. **Maximum Heart Rate**: Strong correlation with heart disease
3. **ST Depression**: Important exercise-related indicator
4. **Number of Major Vessels**: Critical diagnostic feature
5. **Thalassemia**: Genetic factor with high predictive value

### Data Insights
- Men are more likely to have heart disease in this dataset
- Age shows a positive correlation with heart disease risk
- Typical angina pain is more associated with heart disease
- Lower maximum heart rate correlates with higher disease risk

## 📊 Visualizations

The project includes comprehensive visualizations:

- **Correlation Heatmap**: Shows relationships between features
- **Feature Distribution**: Histograms and box plots for each feature
- **Model Performance Comparison**: Bar charts comparing different metrics
- **ROC Curves**: Performance visualization for all models
- **Confusion Matrices**: Detailed prediction accuracy breakdown
- **Feature Importance**: Which features contribute most to predictions

## 🎯 Future Enhancements

- [ ] **Deep Learning Models**: Implement neural networks for improved accuracy
- [ ] **Ensemble Methods**: Combine multiple models for better predictions
- [ ] **Feature Engineering**: Create new features from existing ones
- [ ] **Web Application**: Deploy model as a web service
- [ ] **Real-time Predictions**: API for real-time heart disease prediction
- [ ] **Mobile App**: Mobile application for easy access
- [ ] **Data Augmentation**: Techniques to handle imbalanced datasets
- [ ] **Hyperparameter Tuning**: Advanced optimization techniques

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guide for Python
- Add docstrings to all functions
- Include unit tests for new features
- Update README.md if necessary

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Uday Dixit**
- GitHub: [@dixituday31](https://github.com/dixituday31)
- LinkedIn: [Connect with me](https://linkedin.com/in/udaydixit)
- Email: udayd6269@gmail.com

## 🙏 Acknowledgments

- Heart Disease dataset providers
- UCI Machine Learning Repository
- Scikit-learn community
- Healthcare professionals for domain insights
- Open source contributors

## 📚 References

1. Dua, D. and Graff, C. (2019). UCI Machine Learning Repository. University of California, Irvine, School of Information and Computer Sciences.
2. American Heart Association - Heart Disease Statistics
3. World Health Organization - Cardiovascular Disease Facts
4. Scikit-learn Documentation
5. Python Data Science Handbook

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/dixituday31/Heart-Disease-Prediction/issues) section
2. Create a new issue if your problem isn't already listed
3. Contact the author directly for urgent matters

---

**⭐ If you found this project helpful, please give it a star!**

**💡 Remember: This tool is for educational and research purposes only. Always consult with qualified healthcare professionals for medical advice.**
