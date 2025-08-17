# 🚢 Titanic Passenger Survival Prediction - ML App

A comprehensive machine learning application for predicting passenger survival on the Titanic using Streamlit. This project demonstrates end-to-end ML workflow from data exploration to model deployment.

## 📋 Project Overview

This application provides an interactive interface to:
- **Explore** the Titanic dataset with comprehensive data analysis
- **Visualize** key patterns and relationships in the data
- **Predict** passenger survival using a trained machine learning model
- **Evaluate** model performance with detailed metrics and comparisons

## 🎯 Features

### 🏠 Home Dashboard
- Upload trained model and dataset files
- System status monitoring
- Quick start guide

### 📊 Data Exploration
- Interactive dataset inspection
- Column information and statistics
- Dynamic filtering capabilities
- Sample data preview

### 📈 Visualizations
- Survival rate analysis by gender
- Age distribution across passenger classes
- Fare vs. Age scatter plots with survival coloring
- Interactive chart controls

### 🤖 Prediction Engine
- Real-time survival predictions
- Input validation and error handling
- Confidence scores for predictions
- Feature engineering consistency

### 📐 Model Performance
- Comprehensive evaluation metrics
- Confusion matrix visualization
- Cross-validation comparisons
- Algorithm benchmarking

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn
- **Visualization**: Plotly, Matplotlib
- **Model Persistence**: Joblib
- **Language**: Python 3.7+

## 📦 Installation & Setup

### Prerequisites
- Python 3.7 or higher
- pip package manager

### 1. Clone or Download
```bash
# If using git
git clone <your-repository-url>
cd titanic-survival-prediction

# Or download the files directly to your working directory
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

**Or install manually:**
```bash
pip install streamlit pandas numpy scikit-learn joblib plotly matplotlib
```

### 3. Prepare Your Files
Place the following files in your project directory:
- `app.py` - Main application file
- `train.csv` - Titanic training dataset (from Kaggle)
- `passenger_survival_model.pkl` - Your trained model (from Google Colab)

## 🚀 Usage

### Running the Application
```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

### First Time Setup
1. **Upload Dataset**: Use the file uploader on the Home page to upload your `train.csv` file
2. **Upload Model**: Upload your trained model file (`.pkl` format) from Google Colab
3. **Refresh**: Reload the page to activate the uploaded files

## 📊 Dataset Requirements

Your `train.csv` should contain the standard Titanic features:
- `PassengerId` - Unique passenger identifier
- `Survived` - Target variable (0 = No, 1 = Yes)
- `Pclass` - Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
- `Sex` - Gender (male/female)
- `Age` - Age in years
- `SibSp` - Number of siblings/spouses aboard
- `Parch` - Number of parents/children aboard
- `Fare` - Passenger fare
- `Embarked` - Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)
- `Cabin` - Cabin number (optional, will be dropped)
- `Ticket` - Ticket number (optional, will be dropped)

## 🤖 Model Training (Google Colab)

The model used in this application was trained in Google Colab. Key training steps included:

1. **Data Preprocessing**:
   - Handling missing values (Age, Embarked)
   - Feature engineering (FamilySize, IsAlone)
   - Categorical encoding (Sex, Embarked)

2. **Feature Engineering**:
   - `FamilySize = SibSp + Parch + 1`
   - `IsAlone = 1 if FamilySize == 1 else 0`

3. **Model Training**:
   - Train-test split with stratification
   - Hyperparameter tuning
   - Cross-validation
   - Model persistence using joblib

4. **Export**:
   - Save trained model as `.pkl` file
   - Download to local machine

## 🔧 Customization

### Adding New Models
To use a different model, ensure it has the following methods:
- `predict(X)` - Returns survival predictions
- `predict_proba(X)` - Returns prediction probabilities (optional)

### Feature Engineering
The app automatically applies the same feature engineering used during training:
- Age imputation with median
- Embarked imputation with mode
- Family size calculation
- Alone passenger identification

### Visualization Customization
Modify the visualization section in `app.py` to add new charts or modify existing ones.

## 📈 Performance Metrics

The application evaluates models using:
- **Accuracy** - Overall prediction correctness
- **Precision** - True positive rate among positive predictions
- **Recall** - True positive rate among actual positives
- **F1-Score** - Harmonic mean of precision and recall
- **ROC-AUC** - Area under the receiver operating characteristic curve

## 🐛 Troubleshooting

### Common Issues

1. **Model Not Loading**
   - Ensure the `.pkl` file is in the same directory as `app.py`
   - Check that the model was saved using joblib
   - Verify file permissions

2. **Dataset Errors**
   - Confirm CSV format and encoding
   - Check column names match expected format
   - Ensure no corrupted data

3. **Prediction Failures**
   - Verify feature engineering consistency
   - Check input validation
   - Ensure model was trained with same feature set

### Error Messages
- **"Model not found"** - Upload your trained model file
- **"Dataset not found"** - Upload your training CSV file
- **"Invalid category"** - Check Sex/Embarked values match training data

## 📚 Learning Resources

- [Titanic Dataset on Kaggle](https://www.kaggle.com/datasets/pavlofesenko/titanic-extended?select=train.csv)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

## 🤝 Contributing

Feel free to contribute to this project by:
- Reporting bugs
- Suggesting new features
- Improving documentation
- Adding new visualizations
- Optimizing performance

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

Created as part of a machine learning project using:
- **Dataset Source**: Kaggle
- **Training Platform**: Google Colab
- **Deployment**: Streamlit

---

**Note**: This application is designed for educational and demonstration purposes. The Titanic dataset is historical and should be used for learning machine learning concepts rather than real-world predictions.
