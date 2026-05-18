import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from data_loader import DataLoader
from models import SentimentModel
from evaluator import Evaluator

st.set_page_config(page_title="Sentify Pro", layout="wide", page_icon="🎥")

# Custom CSS for polished look
st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("Sentify Pro 🎥")
st.markdown("### Movie Review Sentiment Analysis")

@st.cache_resource
def load_and_train_models():
    dl = DataLoader()
    try:
        df = dl.load_and_preprocess()
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return None, None, None, None, None
        
    X = df['clean_review']
    y = df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Logistic Regression
    lr_model = SentimentModel('logistic_regression')
    lr_model.train(X_train, y_train)
    
    # Train Naive Bayes
    nb_model = SentimentModel('naive_bayes')
    nb_model.train(X_train, y_train)
    
    return lr_model, nb_model, df, X_test, y_test

with st.spinner("Loading models and data..."):
    lr_model, nb_model, df, X_test, y_test = load_and_train_models()

if df is None:
    st.stop()

# Initialize session state for history
if 'history' not in st.session_state:
    st.session_state.history = []

# Sidebar
st.sidebar.header("Configuration")
model_choice = st.sidebar.selectbox("Select Primary Model", ["Logistic Regression", "Naive Bayes"])
primary_model = lr_model if model_choice == "Logistic Regression" else nb_model

# Tabs
tab1, tab2, tab3 = st.tabs(["Prediction", "Batch Processing", "Evaluation Dashboard"])

def display_prediction(model, text, model_name):
    # Predict
    pred = model.predict([text])[0]
    proba = model.predict_proba([text])[0]
    
    # Get probability for the predicted class
    class_idx = list(model.classes_).index(pred)
    conf = proba[class_idx]
    
    st.markdown(f"<div class='prediction-box'>", unsafe_allow_html=True)
    st.subheader(f"{model_name} Prediction: **{pred.upper()}**")
    st.progress(float(conf), text=f"Confidence: {conf*100:.1f}%")
    st.markdown("</div>", unsafe_allow_html=True)
    
    return pred, conf

with tab1:
    st.header("Single Review Prediction")
    user_input = st.text_area("Type or paste a movie review here:", height=150)
    compare_mode = st.checkbox("Side-by-Side Model Comparison")
    
    if st.button("Predict Sentiment", type="primary"):
        if user_input.strip():
            dl = DataLoader()
            clean_input = dl.clean_text(user_input)
            
            if compare_mode:
                col1, col2 = st.columns(2)
                with col1:
                    pred_lr, conf_lr = display_prediction(lr_model, clean_input, "Logistic Regression")
                with col2:
                    pred_nb, conf_nb = display_prediction(nb_model, clean_input, "Naive Bayes")
                    
                # Note on certainty
                if conf_lr > conf_nb:
                    st.info(f"💡 Logistic Regression is more certain by {(conf_lr - conf_nb)*100:.1f}%.")
                elif conf_nb > conf_lr:
                    st.info(f"💡 Naive Bayes is more certain by {(conf_nb - conf_lr)*100:.1f}%.")
                else:
                    st.info("💡 Both models are equally certain.")
                    
                st.session_state.history.append({
                    "Review": user_input[:50] + "..." if len(user_input) > 50 else user_input,
                    "Model": "Both Models",
                    "Result": f"LR: {pred_lr}, NB: {pred_nb}",
                    "Confidence": f"LR: {conf_lr:.2f}, NB: {conf_nb:.2f}"
                })
            else:
                pred, conf = display_prediction(primary_model, clean_input, model_choice)
                st.session_state.history.append({
                    "Review": user_input[:50] + "..." if len(user_input) > 50 else user_input,
                    "Model": model_choice,
                    "Result": pred,
                    "Confidence": f"{conf:.2f}"
                })
        else:
            st.warning("Please enter a review to predict.")
            
    st.divider()
    st.subheader("Prediction History")
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)
    else:
        st.info("No predictions made yet in this session.")

with tab2:
    st.header("Batch Prediction via CSV")
    st.markdown("Upload a CSV file containing a column named `review`.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        try:
            dl = DataLoader()
            batch_df = dl.prepare_batch(uploaded_file)
            st.success(f"Successfully loaded {len(batch_df)} reviews.")
            
            if st.button("Run Batch Prediction"):
                with st.spinner("Processing..."):
                    # Predict using primary model
                    preds = primary_model.predict(batch_df['clean_review'].tolist())
                    probas = primary_model.predict_proba(batch_df['clean_review'].tolist())
                    
                    confidences = []
                    for i, pred in enumerate(preds):
                        class_idx = list(primary_model.classes_).index(pred)
                        confidences.append(probas[i][class_idx])
                        
                    batch_df['predicted_sentiment'] = preds
                    batch_df['confidence'] = confidences
                    
                    st.dataframe(batch_df[['review', 'predicted_sentiment', 'confidence']].head())
                    
                    csv = batch_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Annotated CSV",
                        data=csv,
                        file_name='annotated_reviews.csv',
                        mime='text/csv',
                    )
        except Exception as e:
            st.error(f"Error processing file: {e}")

with tab3:
    st.header("Model Evaluation Dashboard")
    eval_model_choice = st.selectbox("Select Model to Evaluate", ["Logistic Regression", "Naive Bayes"])
    model_to_eval = lr_model if eval_model_choice == "Logistic Regression" else nb_model
    
    evaluator = Evaluator(model_to_eval, X_test, y_test)
    acc, report = evaluator.get_metrics()
    
    st.subheader(f"Metrics ({eval_model_choice})")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{acc*100:.2f}%")
    if 'weighted avg' in report:
        col2.metric("Precision", f"{report['weighted avg']['precision']:.2f}")
        col3.metric("Recall", f"{report['weighted avg']['recall']:.2f}")
        col4.metric("F1-Score", f"{report['weighted avg']['f1-score']:.2f}")
        
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(evaluator.plot_confusion_matrix())
    with col2:
        roc_fig = evaluator.plot_roc_curve()
        if roc_fig:
            st.pyplot(roc_fig)
        else:
            st.info("ROC Curve not available for this model configuration.")
            
    if eval_model_choice == "Logistic Regression":
        st.subheader("Feature Importance")
        fig_feat = evaluator.plot_feature_importance()
        if fig_feat:
            st.pyplot(fig_feat)
        else:
            st.info("Not enough features for importance plot.")
        
    st.divider()
    st.subheader("Dataset Word Clouds")
    st.pyplot(evaluator.plot_word_clouds(df))
