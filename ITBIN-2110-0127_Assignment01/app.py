import streamlit as st # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
import joblib # type: ignore
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.metrics import ( # type: ignore
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)
import plotly.express as px # type: ignore
import matplotlib.pyplot as plt # type: ignore
from typing import Tuple
import os

st.set_page_config(
    page_title='🚢 Titanic Passenger Survival — ML App', 
    layout='wide'
)

@st.cache_data
def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    if 'Cabin' in df.columns:
        df = df.drop(columns=['Cabin'])
    
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    return df

def build_encoders_from_training(df: pd.DataFrame) -> Tuple[dict, dict]:
    sex_cats = sorted(df['Sex'].dropna().unique().tolist())
    sex_map = {cat: i for i, cat in enumerate(sex_cats)}
    
    emb_cats = sorted(df['Embarked'].dropna().unique().tolist())
    emb_map = {cat: i for i, cat in enumerate(emb_cats)}
    
    return sex_map, emb_map

def encode_inputs(
    pclass: int, 
    sex_str: str, 
    age: float, 
    sibsp: int, 
    parch: int, 
    fare: float, 
    embarked_str: str,
    sex_map: dict, 
    emb_map: dict
):
    if sex_str not in sex_map or embarked_str not in emb_map:
        raise ValueError('Invalid category for Sex or Embarked.')
    
    row = {
        'Pclass': pclass,
        'Sex': sex_map[sex_str],
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'Embarked': emb_map[embarked_str],
        'FamilySize': sibsp + parch + 1,
        'IsAlone': 1 if (sibsp + parch + 1) == 1 else 0
    }
    
    return pd.DataFrame([row])[
        ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize', 'IsAlone']
    ]

@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

def compute_metrics(y_true, y_pred, y_proba=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc = roc_auc_score(y_true, y_proba[:, 1]) if y_proba is not None else np.nan
    
    return acc, prec, rec, f1, roc

# Constants
DEFAULT_DATA = 'train.csv'
DEFAULT_MODEL = 'passenger_survival_model.pkl'

# Load dataset
dataset = None
if os.path.isfile(DEFAULT_DATA):
    with st.spinner('Loading dataset...'):
        dataset = load_dataset(DEFAULT_DATA)

# Load model
model = None
model_load_error = None
if os.path.isfile(DEFAULT_MODEL):
    try:
        with st.spinner('Loading trained model...'):
            model = load_model(DEFAULT_MODEL)
    except Exception as e:
        model_load_error = str(e)

# Sidebar navigation
st.sidebar.header('Navigation')
page = st.sidebar.radio(
    'Go to', 
    ['🏠 Home', '📊 Data Exploration', '📈 Visualisations', '🤖 Prediction', '📐 Model Performance']
)

st.sidebar.markdown('---')
st.sidebar.subheader('Artifacts')
st.sidebar.write(f"**Model file:** `{DEFAULT_MODEL}` " + ('✅' if model is not None else '❌'))
st.sidebar.write(f"**Dataset:** `{DEFAULT_DATA}` " + ('✅' if dataset is not None else '❌'))

# Home page
if page == '🏠 Home':
    st.title('🚢 Titanic Passenger Survival — Prediction & Analysis')
    st.markdown('''
    Welcome! This app lets you:
    - Explore the Titanic training dataset
    - Visualise relationships and distributions
    - Predict passenger **survival** using a trained ML model
    - Review model performance and compare algorithms
    
    **How to use**
    1. Use the sidebar to navigate sections.
    2. If `train.csv` or `passenger_survival_model.pkl` are missing, upload them below.
    ''')

    st.subheader('Upload Artifacts (Optional)')
    c1, c2 = st.columns(2)
    
    with c1:
        up_model = st.file_uploader(
            'Upload trained model (.pkl)', 
            type=['pkl'], 
            help='Upload the joblib/pickle file you trained in Colab.'
        )
        if up_model:
            with open(DEFAULT_MODEL, 'wb') as f:
                f.write(up_model.read())
            st.success('Model uploaded. Please refresh to load.')
    
    with c2:
        up_csv = st.file_uploader(
            'Upload dataset CSV', 
            type=['csv'], 
            help='Upload the Titanic train CSV used for EDA/visualisations.'
        )
        if up_csv:
            df_up = pd.read_csv(up_csv)
            df_up.to_csv(DEFAULT_DATA, index=False)
            st.success('Dataset uploaded. Please refresh to load.')

    if model_load_error:
        st.error(f'Model load error: {model_load_error}')

# Data Exploration page
elif page == '📊 Data Exploration':
    st.title('📊 Data Exploration')
    
    if dataset is None:
        st.warning('Dataset not found. Please upload `train.csv` on the Home page.')
    else:
        df = dataset.copy()
        
        st.subheader('Overview')
        st.write(f'**Shape:** {df.shape[0]} rows × {df.shape[1]} columns')
        
        info_df = pd.DataFrame({
            'Column': df.columns,
            'Non-Null Count': df.notna().sum().values,
            'Dtype': df.dtypes.values
        })
        st.dataframe(info_df, use_container_width=True)

        st.subheader('Sample Data')
        st.dataframe(df.head(10), use_container_width=True)

        st.subheader('Interactive Filter')
        col = st.selectbox('Choose a column to filter', df.columns)
        
        if df[col].dtype == 'object':
            vals = sorted(df[col].dropna().unique().tolist())
            chosen = st.multiselect(
                'Select values', 
                vals, 
                default=vals[:1] if vals else []
            )
            filtered = df[df[col].isin(chosen)] if chosen else df.head(0)
        else:
            min_v, max_v = float(df[col].min()), float(df[col].max())
            rng = st.slider('Select range', min_v, max_v, (min_v, max_v))
            filtered = df[df[col].between(rng[0], rng[1])]
        
        st.caption('Filtered view')
        st.dataframe(filtered.head(200), use_container_width=True)

# Visualisations page
elif page == '📈 Visualisations':
    st.title('📈 Visualisations')
    
    if dataset is None:
        st.warning('Dataset not found. Please upload `train.csv` on the Home page.')
    else:
        df = preprocess_df(dataset)

        st.subheader('1) Survival Rate by Sex')
        sex_survival = df.groupby('Sex')['Survived'].mean().reset_index()
        
        if df['Sex'].dtype != object:
            sex_survival['Sex'] = sex_survival['Sex'].map(
                {0: 'female', 1: 'male'}
            ).fillna(sex_survival['Sex'])
        
        fig1 = px.bar(
            sex_survival, 
            x='Sex', 
            y='Survived', 
            labels={'Survived': 'Mean Survival'}
        )
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader('2) Age Distribution (filter by Pclass)')
        pclass_sel = st.multiselect(
            'Select Pclass', 
            sorted(df['Pclass'].unique().tolist()), 
            default=sorted(df['Pclass'].unique().tolist())
        )
        df_age = df[df['Pclass'].isin(pclass_sel)]
        fig2 = px.histogram(
            df_age, 
            x='Age', 
            nbins=30, 
            color='Pclass', 
            barmode='overlay'
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader('3) Fare vs Age (color = Survived)')
        fig3 = px.scatter(
            df, 
            x='Age', 
            y='Fare', 
            color=df['Survived'].map({0: 'No', 1: 'Yes'}), 
            labels={'color': 'Survived'}
        )
        st.plotly_chart(fig3, use_container_width=True)

# Prediction page
elif page == '🤖 Prediction':
    st.title('🤖 Passenger Survival Prediction')
    
    if model is None:
        st.warning('Model not found. Please upload `passenger_survival_model.pkl` on the Home page.')
    else:
        if dataset is None:
            st.warning('Dataset not found. Upload `train.csv` to ensure category encodings match training.')
        else:
            df = preprocess_df(dataset)
            sex_map, emb_map = build_encoders_from_training(df)

            st.markdown('Enter passenger details to get a prediction:')
            c1, c2, c3 = st.columns(3)
            
            with c1:
                pclass = st.selectbox(
                    'Passenger Class (1=1st, 2=2nd, 3=3rd)', 
                    [1, 2, 3], 
                    help='Ticket class'
                )
                sex = st.selectbox('Sex', sorted(sex_map.keys()))
                age = st.slider('Age', 0, 80, 29)
            
            with c2:
                sibsp = st.number_input(
                    'Siblings/Spouses aboard (SibSp)', 
                    min_value=0, 
                    max_value=10, 
                    value=0
                )
                parch = st.number_input(
                    'Parents/Children aboard (Parch)', 
                    min_value=0, 
                    max_value=10, 
                    value=0
                )
                fare = st.number_input(
                    'Fare', 
                    min_value=0.0, 
                    max_value=600.0, 
                    value=32.2
                )
            
            with c3:
                embarked = st.selectbox(
                    'Port of Embarkation (Embarked)', 
                    sorted(emb_map.keys())
                )

            # Validation
            error_msgs = []
            if age is None or age < 0 or age > 120:
                error_msgs.append('Age must be between 0 and 120.')
            if fare is None or fare < 0:
                error_msgs.append('Fare must be non-negative.')

            if error_msgs:
                for m in error_msgs:
                    st.error(m)
            else:
                with st.spinner('Predicting...'):
                    try:
                        X_input = encode_inputs(
                            pclass, sex, age, sibsp, parch, fare, embarked, sex_map, emb_map
                        )
                        pred = model.predict(X_input)[0]
                        proba = model.predict_proba(X_input)[0] if hasattr(model, 'predict_proba') else None
                        
                        label = '✅ Survived' if pred == 1 else '❌ Did not survive'
                        st.success(f'**Prediction:** {label}')
                        
                        if proba is not None:
                            st.info(f'**Confidence:** {np.max(proba):.2%}')
                        
                        st.caption('Prediction uses the same feature engineering as training (FamilySize, IsAlone).')
                    except Exception as e:
                        st.error(f'Prediction failed: {e}')

# Model Performance page
elif page == '📐 Model Performance':
    st.title('📐 Model Performance & Comparison')
    
    if dataset is None or model is None:
        st.warning('Need both dataset (`train.csv`) and model (`passenger_survival_model.pkl`). Upload them on Home.')
    else:
        df = preprocess_df(dataset)
        features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize', 'IsAlone']
        sex_map, emb_map = build_encoders_from_training(df)
        
        df_enc = df.copy()
        df_enc['Sex'] = df_enc['Sex'].map(sex_map)
        df_enc['Embarked'] = df_enc['Embarked'].map(emb_map)
        
        X = df_enc[features]
        y = df_enc['Survived']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        with st.spinner('Evaluating loaded model...'):
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            acc, prec, rec, f1, roc = compute_metrics(y_test, y_pred, y_proba)
        
        # Display metrics
        mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
        mcol1.metric('Accuracy', f'{acc:.3f}')
        mcol2.metric('Precision', f'{prec:.3f}')
        mcol3.metric('Recall', f'{rec:.3f}')
        mcol4.metric('F1-score', f'{f1:.3f}')
        mcol5.metric('ROC-AUC', f'{roc:.3f}' if not np.isnan(roc) else 'N/A')

        st.subheader('Confusion Matrix')
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        
        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, int(val), ha='center', va='center')
        
        st.pyplot(fig, use_container_width=True)

        st.subheader('Model Comparison (5-fold CV on training set)')
        with st.spinner('Running cross-validation...'):
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            logreg = LogisticRegression(max_iter=1000)
            rf = RandomForestClassifier(n_estimators=200, random_state=42)
            
            logreg_scores = cross_val_score(logreg, X_train, y_train, cv=cv, scoring='accuracy')
            rf_scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring='accuracy')
        
        comp_df = pd.DataFrame({
            'Model': ['Logistic Regression', 'Random Forest'],
            'CV Mean Accuracy': [logreg_scores.mean(), rf_scores.mean()]
        })
        
        st.dataframe(comp_df.round(4), use_container_width=True)
        
        figb = px.bar(
            comp_df, 
            x='Model', 
            y='CV Mean Accuracy', 
            text='CV Mean Accuracy', 
            range_y=[0, 1]
        )
        st.plotly_chart(figb, use_container_width=True)
        
        st.caption('Note: CV comparison re-trains simple baselines on the training set for a fair side-by-side view.')

# Footer
st.markdown('---')
with st.expander('ℹ️ Help & Documentation'):
    st.markdown('''
    **Sections**
    - **Home**: Upload or confirm model & dataset files.
    - **Data Exploration**: Inspect shape, columns, types; filter the dataset interactively.
    - **Visualisations**: Explore survival patterns via interactive charts.
    - **Prediction**: Enter passenger details and get a survival prediction with confidence.
    - **Model Performance**: Evaluate metrics on a holdout set, view confusion matrix, and compare algorithms via cross-validation.
    
    **Tips**
    - Ensure the dataset columns are the standard Titanic fields: `Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, Survived`.
    - The app auto-engineers **FamilySize** and **IsAlone** to match training.
    - If predictions fail, check that the uploaded model was trained with the same feature order.
    ''')