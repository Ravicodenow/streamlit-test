#!/usr/bin/env python3
"""
Credit Card Fraud Detection - Single File Streamlit App
========================================================

A complete credit card fraud detection application using Artificial Neural Networks.
This single file contains the entire application including:
- ML model training and prediction
- Interactive Streamlit web interface
- Data visualization and analysis
- Bulk transaction processing

Author: AI Assistant
Date: October 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.datasets import make_classification
import warnings
warnings.filterwarnings('ignore')

# Page configuration
# User for page layout
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .fraud-alert {
        background-color: #ff5252;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #d32f2f;
        box-shadow: 0 4px 8px rgba(255, 82, 82, 0.3);
        font-weight: bold;
    }
    .safe-alert {
        background-color: #4caf50;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #388e3c;
        box-shadow: 0 4px 8px rgba(76, 175, 80, 0.3);
        font-weight: bold;
    }
    .training-info {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #2196f3;
        margin: 1rem 0;
    }
    .transaction-details {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.8rem;
        border: 1px solid #e9ecef;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .feature-stats {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Backend model and processing class
class CreditCardFraudDetector:
    """Credit Card Fraud Detection using Artificial Neural Network"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.model_info = {}
        self.expected_features = None  # Track expected number of features
        self.feature_columns = None   # Track feature column names
        
    def create_sample_dataset(self, n_samples=10000, n_features=30):
        """Create a sample credit card fraud dataset for testing"""
        st.info(f"🔄 Creating sample dataset with {n_features} features...")
        
        # Generate synthetic dataset with flexible feature count
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=min(n_features-2, 20),  # Adjust informative features
            n_redundant=min(n_features//6, 5),    # Adjust redundant features
            n_clusters_per_class=1,
            weights=[0.99, 0.01],  # Imbalanced like real fraud data
            flip_y=0.01,
            random_state=42
        )
        
        # Create feature names dynamically
        if n_features <= 30:
            # Standard format: Time + V-features + Amount
            v_count = n_features - 2
            feature_names = ['Time'] + [f'V{i}' for i in range(1, v_count + 1)] + ['Amount']
        else:
            # Extended format: Time + more V-features + Amount
            v_count = n_features - 2
            feature_names = ['Time'] + [f'V{i}' for i in range(1, v_count + 1)] + ['Amount']
        
        # Create DataFrame
        df = pd.DataFrame(X, columns=feature_names)
        
        # Adjust Time and Amount to be more realistic
        df['Time'] = np.random.uniform(0, 172800, size=len(df))  # 48 hours in seconds
        df['Amount'] = np.random.exponential(88.3, size=len(df))  # Realistic transaction amounts
        
        # Add target
        df['Class'] = y
        
        return df
    
    def load_and_preprocess_data(self, data_source="sample"):
        """Load and preprocess the credit card dataset"""
        
        if data_source == "sample":
            df = self.create_sample_dataset()
        elif isinstance(data_source, pd.DataFrame):
            df = data_source.copy()
        else:
            # Try to load from file
            try:
                df = pd.read_csv(data_source)
            except Exception as e:
                st.error(f"Error loading dataset: {e}")
                return None, None
        
        # Basic data info
        fraud_rate = df['Class'].mean() * 100
        st.success(f"📊 Dataset loaded: {df.shape[0]:,} transactions, {df['Class'].sum():,} fraud cases ({fraud_rate:.2f}%)")
        
        # Separate features and target
        X = df.drop('Class', axis=1)
        y = df['Class']
        
        # Store feature information for consistency
        self.expected_features = X.shape[1]
        self.feature_columns = X.columns.tolist()
        
        return X, y
    
    def build_ann_model(self, input_dim):
        """Build Artificial Neural Network for binary classification with dynamic architecture"""
        
        # Dynamic architecture based on input dimensions
        if input_dim <= 10:
            hidden_layers = (32, 16)
        elif input_dim <= 30:
            hidden_layers = (64, 32, 16)
        elif input_dim <= 50:
            hidden_layers = (128, 64, 32)
        else:
            hidden_layers = (256, 128, 64, 32)
        
        model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation='relu',
            solver='adam',
            alpha=0.001,  # L2 regularization
            batch_size=min(256, max(32, input_dim * 8)),  # Dynamic batch size
            learning_rate='adaptive',
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10
        )
        return model
    
    def train_model(self, data_source="sample", test_size=0.2):
        """Train the ANN model"""
        # Load and preprocess data
        X, y = self.load_and_preprocess_data(data_source)
        
        if X is None or y is None:
            return False
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Build the model
        self.model = self.build_ann_model(X_train_scaled.shape[1])
        
        # Calculate class weights for imbalanced dataset
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, class_weights))
        
        # Training progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🧠 Training Neural Network...")
        progress_bar.progress(25)
        
        # Train the model
        self.model.fit(X_train_scaled, y_train, sample_weight=class_weights[y_train])
        
        progress_bar.progress(75)
        status_text.text("📊 Evaluating model performance...")
        
        # Evaluate the model
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Store model info
        self.model_info = {
            'roc_auc': roc_auc,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'fraud_rate': y.mean(),
            'class_weights': class_weight_dict
        }
        
        progress_bar.progress(100)
        status_text.text("✅ Training completed!")
        
        self.is_trained = True
        
        # Display results
        st.success(f"🎯 Model trained successfully! ROC-AUC Score: {roc_auc:.4f}")
        
        return True
    
    def predict(self, transaction_data):
        """Predict fraud probability for new transaction with flexible input"""
        if not self.is_trained:
            raise ValueError("Model is not trained!")
        
        # Ensure input is numpy array
        if isinstance(transaction_data, list):
            transaction_data = np.array(transaction_data).reshape(1, -1)
        elif len(transaction_data.shape) == 1:
            transaction_data = transaction_data.reshape(1, -1)
        
        # Check feature count consistency
        input_features = transaction_data.shape[1]
        if input_features != self.expected_features:
            st.warning(f"⚠️ Input has {input_features} features, but model expects {self.expected_features}")
            
            # Try to adapt the input
            if input_features > self.expected_features:
                # Too many features - take the first N
                transaction_data = transaction_data[:, :self.expected_features]
                st.info(f"🔧 Using first {self.expected_features} features")
            else:
                # Too few features - pad with zeros
                padding = np.zeros((1, self.expected_features - input_features))
                transaction_data = np.concatenate([transaction_data, padding], axis=1)
                st.info(f"🔧 Padded with zeros to match {self.expected_features} features")
        
        # Scale the input
        transaction_scaled = self.scaler.transform(transaction_data)
        
        # Predict
        fraud_probability = self.model.predict_proba(transaction_scaled)[0][1]
        prediction = 1 if fraud_probability > 0.5 else 0
        
        return {
            'prediction': int(prediction),
            'fraud_probability': float(fraud_probability),
            'is_fraud': bool(prediction)
        }
    
    def predict_bulk(self, transactions_data):
        """Predict fraud for multiple transactions with flexible input"""
        if not self.is_trained:
            raise ValueError("Model is not trained!")
        
        # Convert to numpy array
        transaction_array = np.array(transactions_data)
        
        # Check feature count consistency
        input_features = transaction_array.shape[1]
        if input_features != self.expected_features:
            st.warning(f"⚠️ Input has {input_features} features, but model expects {self.expected_features}")
            
            # Try to adapt the input
            if input_features > self.expected_features:
                # Too many features - take the first N
                transaction_array = transaction_array[:, :self.expected_features]
                st.info(f"🔧 Using first {self.expected_features} features for all transactions")
            else:
                # Too few features - pad with zeros
                padding = np.zeros((transaction_array.shape[0], self.expected_features - input_features))
                transaction_array = np.concatenate([transaction_array, padding], axis=1)
                st.info(f"🔧 Padded all transactions with zeros to match {self.expected_features} features")
        
        # Scale the data
        transaction_scaled = self.scaler.transform(transaction_array)
        
        # Batch prediction
        fraud_probabilities = self.model.predict_proba(transaction_scaled)[:, 1]
        predictions = (fraud_probabilities > 0.5).astype(int)
        
        results = []
        for i, (prob, pred) in enumerate(zip(fraud_probabilities, predictions)):
            results.append({
                'prediction': int(pred),
                'fraud_probability': float(prob),
                'is_fraud': bool(pred)
            })
        
        return results

# Frontend with Streamlit
@st.cache_resource
def get_fraud_detector():
    """Get or create the fraud detector instance"""
    return CreditCardFraudDetector()

def generate_sample_transaction():
    """Generate a sample transaction with realistic values"""
    # Generate 28 anonymized features (V1-V28) with realistic distributions
    v_features = np.random.normal(0, 1, 28).tolist()
    
    # Time feature (seconds elapsed since first transaction)
    time_feature = np.random.uniform(0, 172800)  # 48 hours
    
    # Amount feature (transaction amount)
    amount_feature = np.random.exponential(88.3)  # Based on real dataset distribution
    
    return [time_feature] + v_features + [amount_feature]

def display_prediction_result(result):
    """Display prediction results with visualizations"""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if result['is_fraud']:
            st.markdown(f"""
            <div class="fraud-alert">
                <h3>🚨 FRAUD DETECTED</h3>
                <p><strong>Confidence:</strong> {result['fraud_probability']:.2%}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="safe-alert">
                <h3>✅ LEGITIMATE</h3>
                <p><strong>Confidence:</strong> {(1-result['fraud_probability']):.2%}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Probability gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = result['fraud_probability'] * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Fraud Probability (%)"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 25], 'color': "lightgreen"},
                    {'range': [25, 75], 'color': "yellow"},
                    {'range': [75, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        # Risk assessment
        risk_level = "HIGH" if result['fraud_probability'] > 0.7 else "MEDIUM" if result['fraud_probability'] > 0.3 else "LOW"
        st.metric("Risk Level", risk_level)
        st.metric("Prediction", "FRAUD" if result['is_fraud'] else "LEGITIMATE")

def analyze_bulk_transactions(df, detector):
    """Analyze multiple transactions using the trained model"""
    
    # Convert DataFrame to list of lists
    transactions = df.values.tolist()
    total_transactions = len(transactions)
    
    st.info(f"🚀 Processing {total_transactions:,} transactions...")
    
    # Process in batches for better performance
    batch_size = 1000
    all_results = []
    overall_fraud_count = 0
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(0, total_transactions, batch_size):
        batch_end = min(i + batch_size, total_transactions)
        batch_transactions = transactions[i:batch_end]
        
        status_text.text(f"Processing batch {i//batch_size + 1} ({i+1:,}-{batch_end:,} of {total_transactions:,})...")
        
        # Make bulk prediction
        try:
            results = detector.predict_bulk(batch_transactions)
            all_results.extend(results)
            
            # Count frauds in this batch
            batch_fraud_count = sum(1 for r in results if r['is_fraud'])
            overall_fraud_count += batch_fraud_count
            
        except Exception as e:
            st.error(f"Error processing batch {i//batch_size + 1}: {e}")
            break
        
        # Update progress
        progress_bar.progress(batch_end / total_transactions)
    
    status_text.text("✅ Processing complete!")
    
    if all_results:
        # Overall statistics
        fraud_rate = overall_fraud_count / total_transactions
        avg_prob = np.mean([r['fraud_probability'] for r in all_results])
        
        # Display summary
        st.markdown("## 📊 Analysis Results")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Transactions", f"{total_transactions:,}")
        with col2:
            st.metric("🚨 Fraud Detected", f"{overall_fraud_count:,}")
        with col3:
            st.metric("📈 Fraud Rate", f"{fraud_rate:.2%}")
        with col4:
            st.metric("📊 Avg Fraud Probability", f"{avg_prob:.2%}")
        
        # Risk distribution
        st.markdown("### 🎯 Risk Distribution")
        
        # Categorize by risk level
        high_risk = sum(1 for r in all_results if r['fraud_probability'] > 0.7)
        medium_risk = sum(1 for r in all_results if 0.3 < r['fraud_probability'] <= 0.7)
        low_risk = sum(1 for r in all_results if r['fraud_probability'] <= 0.3)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🔴 High Risk (>70%)", f"{high_risk:,}", f"{high_risk/total_transactions:.1%}")
        with col2:
            st.metric("🟡 Medium Risk (30-70%)", f"{medium_risk:,}", f"{medium_risk/total_transactions:.1%}")
        with col3:
            st.metric("🟢 Low Risk (<30%)", f"{low_risk:,}", f"{low_risk/total_transactions:.1%}")
        
        # Visualizations
        st.markdown("### 📈 Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Fraud probability distribution
            probs = [r['fraud_probability'] for r in all_results]
            fig = px.histogram(
                x=probs,
                nbins=50,
                title="Distribution of Fraud Probabilities",
                labels={'x': 'Fraud Probability', 'y': 'Count'},
                color_discrete_sequence=['steelblue']
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Risk level pie chart
            risk_data = {
                'Risk Level': ['High Risk', 'Medium Risk', 'Low Risk'],
                'Count': [high_risk, medium_risk, low_risk],
                'Color': ['#ff4444', '#ffaa00', '#44ff44']
            }
            
            fig = px.pie(
                values=risk_data['Count'],
                names=risk_data['Risk Level'],
                title="Risk Level Distribution",
                color_discrete_sequence=risk_data['Color']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Download results option
        if st.button("📥 Download Results as CSV"):
            results_df = pd.DataFrame([
                {
                    'Transaction_ID': i+1,
                    'Prediction': r['prediction'],
                    'Fraud_Probability': r['fraud_probability'],
                    'Is_Fraud': r['is_fraud'],
                    'Risk_Level': ('High' if r['fraud_probability'] > 0.7 
                                 else 'Medium' if r['fraud_probability'] > 0.3 
                                 else 'Low')
                }
                for i, r in enumerate(all_results)
            ])
            
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"fraud_analysis_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">💳 Credit Card Fraud Detection</h1>', unsafe_allow_html=True)
    st.markdown("### 🧠 Powered by Artificial Neural Networks")
    
    # Get detector instance
    detector = get_fraud_detector()
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Model status
        if detector.is_trained:
            st.success("✅ Model Ready")
            if hasattr(detector, 'model_info') and detector.model_info:
                st.info(f"🎯 ROC-AUC: {detector.model_info.get('roc_auc', 0):.4f}")
        else:
            st.warning("⚠️ Model Not Trained")
        
        st.markdown("---")
        
        # Train model section
        st.subheader("🏋️ Model Training")
        
        data_source = st.radio(
            "Choose training data:",
            ["Sample Dataset (Demo)", "Upload CSV File"]
        )
        
        if data_source == "Upload CSV File":
            uploaded_file = st.file_uploader(
                "Upload training dataset",
                type=['csv'],
                help="CSV should have 30 feature columns and 1 'Class' target column"
            )
        
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("Training model..."):
                if data_source == "Sample Dataset (Demo)":
                    success = detector.train_model("sample")
                else:
                    if uploaded_file is not None:
                        try:
                            train_df = pd.read_csv(uploaded_file)
                            if 'Class' not in train_df.columns:
                                st.error("Dataset must have a 'Class' column for training")
                                success = False
                            else:
                                success = detector.train_model(train_df)
                        except Exception as e:
                            st.error(f"Error loading training file: {e}")
                            success = False
                    else:
                        st.error("Please upload a CSV file for training")
                        success = False
                
                if success:
                    st.rerun()
        
        st.markdown("---")
        st.markdown("### 📊 Model Architecture")
        st.info("""
        **ANN Configuration:**
        - Input Layer: 30 features
        - Hidden Layers: 64 → 32 → 16 neurons
        - Activation: ReLU
        - Output: Sigmoid (fraud probability)
        - Regularization: L2 + Early Stopping
        """)
    
    # Main content area
    if not detector.is_trained:
        st.markdown("""
        <div class="training-info">
            <h3>🚀 Welcome to Credit Card Fraud Detection!</h3>
            <p>To get started, please train the model using the sidebar controls.</p>
            <p><strong>Options:</strong></p>
            <ul>
                <li><strong>Sample Dataset:</strong> Use synthetic data for quick demo</li>
                <li><strong>Upload CSV:</strong> Train with your own dataset</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Show sample data format
        st.markdown("### 📋 Expected Data Format")
        sample_data = {
            'Time': [0, 1, 2],
            'V1': [-1.36, 1.19, -1.36],
            'V2': [-0.07, 0.27, -0.07],
            '...': ['...', '...', '...'],
            'V28': [-0.02, 0.01, -0.02],
            'Amount': [149.62, 2.69, 149.62],
            'Class': [0, 0, 1]
        }
        st.dataframe(pd.DataFrame(sample_data), use_container_width=True)
        st.caption("30 features (Time, V1-V28, Amount) + 1 target (Class: 0=Legitimate, 1=Fraud)")
        
        return
    
    # Input method selection
    st.markdown("---")
    input_method = st.radio(
        "🔍 Choose Analysis Method:",
        ["Manual Input", "Sample Transaction", "Bulk CSV Upload"],
        horizontal=True
    )
    
    if input_method == "Manual Input":
        st.header("🔢 Manual Feature Input")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Transaction Details")
            time_val = st.number_input("Time (seconds since first transaction)", 
                                     value=0.0, min_value=0.0)
            amount_val = st.number_input("Amount ($)", 
                                       value=100.0, min_value=0.0)
        
        with col2:
            st.subheader("Anonymized Features (V1-V28)")
            st.info("These are PCA-transformed features from the original dataset")
        
        # Create input fields for V1-V28 features
        v_features = []
        cols = st.columns(4)
        for i in range(28):
            with cols[i % 4]:
                v_val = st.number_input(f"V{i+1}", value=0.0, 
                                      key=f"v{i+1}", format="%.6f")
                v_features.append(v_val)
        
        features = [time_val] + v_features + [amount_val]
        
        if st.button("🔍 Predict Fraud", type="primary"):
            with st.spinner("Analyzing transaction..."):
                try:
                    result = detector.predict(features)
                    display_prediction_result(result)
                except Exception as e:
                    st.error(f"Prediction error: {e}")
    
    elif input_method == "Sample Transaction":
        st.header("🎲 Sample Transaction Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.info("Click the button below to generate and analyze a random transaction")
        
        with col2:
            generate_clicked = st.button("🎯 Generate & Predict", type="primary")
        
        # Display results in full width below the button section
        if generate_clicked:
            features = generate_sample_transaction()
            
            # Display transaction details in full width with better styling
            st.markdown("---")
            
            # Use container for better organization
            with st.container():
                st.markdown('<div class="transaction-details">', unsafe_allow_html=True)
                st.subheader("📊 Generated Transaction Details")
                
                # Create a better layout for transaction details
                col_metrics, col_table = st.columns([1, 2])
                
                with col_metrics:
                    st.markdown('<div class="feature-stats">', unsafe_allow_html=True)
                    st.markdown("#### 💰 Key Metrics")
                    st.metric("⏰ Time", f"{features[0]:.2f} seconds", 
                             help="Seconds elapsed since the first transaction in the dataset")
                    st.metric("💵 Amount", f"${features[-1]:.2f}", 
                             help="Transaction amount in USD")
                    
                    # Calculate some basic statistics
                    v_features_only = features[1:29]
                    st.metric("📊 Avg V-Features", f"{np.mean(v_features_only):.3f}",
                             help="Average of all PCA-transformed features")
                    st.metric("📈 Max V-Feature", f"{max(v_features_only):.3f}",
                             help="Highest PCA-transformed feature value")
                    st.metric("📉 Min V-Feature", f"{min(v_features_only):.3f}",
                             help="Lowest PCA-transformed feature value")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col_table:
                    st.markdown("#### 🔢 All Features")
                    # Create a more organized DataFrame with better formatting
                    df_features = pd.DataFrame({
                        'Feature': ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)],
                        'Value': [f"{features[0]:.2f}", f"{features[-1]:.2f}"] + [f"{v:.6f}" for v in features[1:29]],
                        'Type': ['Temporal', 'Monetary'] + ['PCA-Transformed'] * 28
                    })
                    
                    # Add search functionality for the table
                    st.caption("🔍 You can scroll through all 30 features below:")
                    st.dataframe(
                        df_features, 
                        height=350, 
                        use_container_width=True,
                        column_config={
                            "Feature": st.column_config.TextColumn("Feature", width="medium"),
                            "Value": st.column_config.NumberColumn("Value", format="%.6f"),
                            "Type": st.column_config.TextColumn("Type", width="medium")
                        }
                    )
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Make prediction with better spacing and styling
            st.markdown("---")
            st.subheader("🎯 Fraud Analysis Results")
            
            with st.spinner("🔍 Analyzing transaction for fraud patterns..."):
                try:
                    result = detector.predict(features)
                    # Display prediction results in full width with enhanced styling
                    st.success("✅ Analysis completed successfully!")
                    display_prediction_result(result)
                except Exception as e:
                    st.error(f"❌ Prediction error: {e}")
                    st.info("💡 **Tip:** Try training the model first using the sidebar if you haven't already.")
    
    else:  # Bulk CSV Upload
        st.header("📄 Bulk CSV Analysis")
        
        # Add data format explanation and PCA description
        st.markdown("### 📋 Expected Data Format")
        
        # Create tabs for better organization
        tab1, tab2 = st.tabs(["📊 Data Format", "🔬 About PCA Features"])
        
        with tab1:
            st.markdown("""
            ### 📁 What type of data file do I need?
            
            You need a **CSV file with credit card transaction data**. Don't worry about the exact format - our system will help validate it!
            """)
            
            # Create a more user-friendly explanation
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.info("""
                **🎯 Your CSV should include:**
                
                🕐 **Time column** - Transaction timing  
                💰 **Amount column** - Transaction value  
                🔐 **Security features** - Anonymous pattern data (V1, V2, V3, etc.)  
                📊 **Optional: Known results** - If you have fraud labels
                
                **Most datasets have 30-31 columns total**
                """)
                
                st.success("""
                **✅ Common Formats We Accept:**
                
                • Standard fraud detection datasets  
                • Kaggle credit card data  
                • Bank transaction exports (anonymized)  
                • Research datasets with PCA features  
                
                **Just upload and we'll check the format for you!**
                """)
            
            with col2:
                st.markdown("#### 📊 What your data might look like:")
                # Show sample data with more user-friendly values
                sample_data = pd.DataFrame({
                    'Time': [0, 406, 540],
                    'V1': [-1.36, 1.19, -3.04],
                    'V2': [-0.07, 0.27, 1.01],
                    'V3': [2.54, 0.17, -3.16],
                    '...': ['28 more V columns...', '...', '...'],
                    'Amount': [149.62, 2.69, 378.66],
                    'Class': ['0 (Normal)', '0 (Normal)', '1 (Fraud)']
                })
                
                st.dataframe(sample_data, use_container_width=True)
                st.caption("� Don't worry about exact column names - we'll guide you!")
            
            # Add helpful guidance
            st.markdown("---")
            st.markdown("#### 🚀 Getting Started:")
            
            tip_col1, tip_col2, tip_col3 = st.columns(3)
            
            with tip_col1:
                st.markdown("""
                **📋 File Requirements**
                - CSV format (.csv file)
                - Column headers in first row
                - Numeric data (no text values)
                - No missing/empty cells
                """)
            
            with tip_col2:
                st.markdown("""
                **🔢 Typical Data**
                - 30+ columns with transaction features
                - Time values (any format)
                - Amount in dollars/currency
                - V1-V28 security features (-5 to +5 range)
                """)
            
            with tip_col3:
                st.markdown("""
                **❓ Need Help?**
                - Try our sample transaction generator
                - Check the PCA features explanation
                - Upload any credit card dataset to test
                - We'll validate and guide you
                """)
        
        with tab2:
            st.markdown("""
            ### 🔬 What are PCA Features (V1-V28)?
            
            **Principal Component Analysis (PCA)** is a dimensionality reduction technique used to:
            
            #### 🔒 **Privacy Protection**
            - Original credit card features contain sensitive information (card numbers, merchant details, etc.)
            - PCA transforms these into anonymized mathematical representations
            - Protects customer privacy while preserving fraud detection patterns
            
            #### 📊 **How PCA Works**
            1. **Original Features**: Real transaction data (sensitive)
            2. **Mathematical Transformation**: Convert to uncorrelated components
            3. **V1-V28**: The most important 28 components that capture fraud patterns
            4. **Result**: Anonymized features that maintain predictive power
            
            #### 💡 **Key Points**
            - **V1-V28** are the result of PCA transformation of confidential features
            - Each V-feature is a linear combination of original features
            - **Time** and **Amount** are kept in original form (less sensitive)
            - Values typically range from -5 to +5 (standardized)
            - Higher absolute values may indicate unusual patterns
            
            #### 🎯 **For Your Data**
            - If you have raw transaction data, you'll need to apply PCA transformation first
            - If you already have V1-V28 features, you can use them directly
            - The model expects exactly 28 V-features for optimal performance
            """)
            
            # Add a visual representation
            col1, col2 = st.columns(2)
            with col1:
                st.info("""
                **✅ Good V-Feature Values:**
                - V1: -1.35 (normal range)
                - V2: 0.26 (typical value)
                - V3: 2.53 (acceptable)
                """)
            
            with col2:
                st.warning("""
                **⚠️ Suspicious V-Feature Values:**
                - V15: -8.42 (extreme outlier)
                - V22: 15.73 (very unusual)
                - V7: -12.11 (potential fraud indicator)
                """)
        
        st.markdown("---")
        
        uploaded_file = st.file_uploader(
            "Upload CSV file with transaction data",
            type=['csv'],
            help="Upload any credit card dataset - we'll automatically detect and adapt to your format!"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                st.success(f"✅ Loaded {len(df):,} transactions from your CSV file")
                
                # Smart data format detection and validation
                st.markdown("### 🔍 Smart Data Analysis")
                
                original_columns = df.shape[1]
                st.info(f"📊 Analyzing your dataset: {len(df):,} transactions with {original_columns} columns")
                
                # Smart column detection
                def detect_and_prepare_data(df):
                    """Intelligently detect and prepare data for prediction"""
                    
                    # Look for common column patterns
                    cols = df.columns.tolist()
                    
                    # Find Time column (various names)
                    time_cols = [col for col in cols if 'time' in col.lower() or 'timestamp' in col.lower()]
                    
                    # Find Amount column (various names)  
                    amount_cols = [col for col in cols if 'amount' in col.lower() or 'value' in col.lower() or 'transaction' in col.lower()]
                    
                    # Find V-features (PCA columns)
                    v_cols = [col for col in cols if col.startswith('V') and col[1:].isdigit()]
                    
                    # Find Class/Target column
                    class_cols = [col for col in cols if col.lower() in ['class', 'target', 'label', 'fraud', 'is_fraud']]
                    
                    st.write("🔍 **Column Detection Results:**")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if time_cols:
                            st.success(f"⏰ Time column: {time_cols[0]}")
                            detected_time = time_cols[0]
                        else:
                            # Try to find first numeric column that could be time
                            numeric_cols = df.select_dtypes(include=[np.number]).columns
                            if len(numeric_cols) > 0:
                                detected_time = numeric_cols[0]
                                st.warning(f"⏰ Using '{detected_time}' as time column")
                            else:
                                st.error("❌ No time column found")
                                return None
                    
                    with col2:
                        if amount_cols:
                            st.success(f"💰 Amount column: {amount_cols[0]}")
                            detected_amount = amount_cols[0]
                        else:
                            # Try to find a column that looks like amounts
                            numeric_cols = df.select_dtypes(include=[np.number]).columns
                            potential_amount = [col for col in numeric_cols if col != detected_time]
                            if potential_amount:
                                detected_amount = potential_amount[-1]  # Often the last numeric column
                                st.warning(f"💰 Using '{detected_amount}' as amount column")
                            else:
                                st.error("❌ No amount column found")
                                return None
                    
                    with col3:
                        if v_cols:
                            st.success(f"🔢 Found {len(v_cols)} V-features")
                        else:
                            # Look for numeric columns that could be features
                            all_numeric = df.select_dtypes(include=[np.number]).columns
                            feature_cols = [col for col in all_numeric if col not in [detected_time, detected_amount]]
                            st.warning(f"🔢 Using {len(feature_cols)} feature columns")
                            # Rename them to V1, V2, etc. for consistency
                            v_cols = [f'V{i+1}' for i in range(len(feature_cols))]
                    
                    # Handle class column if present
                    target_removed = False
                    if class_cols:
                        class_col = class_cols[0]
                        st.info(f"🏷️ Found target column '{class_col}' - will remove for prediction")
                        target_values = df[class_col].copy()
                        df = df.drop(class_col, axis=1)
                        target_removed = True
                        
                        # Show class distribution
                        if target_values.dtype == 'object':
                            fraud_count = sum(target_values.str.lower().isin(['fraud', '1', 'true', 'yes']))
                        else:
                            fraud_count = sum(target_values == 1)
                        st.success(f"📊 Found {fraud_count:,} fraud cases ({fraud_count/len(target_values)*100:.2f}% fraud rate)")
                    
                    # Reconstruct dataframe with standard format
                    try:
                        # Prepare the final dataset
                        final_df = pd.DataFrame()
                        
                        # Add time column
                        final_df['Time'] = df[detected_time]
                        
                        # Add V-features (either existing V1-V28 or rename other features)
                        if len(v_cols) > 0 and all(col in df.columns for col in v_cols):
                            # We have proper V-columns
                            for v_col in sorted(v_cols):
                                final_df[v_col] = df[v_col]
                        else:
                            # Use other numeric features as V-columns
                            feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                                          if col not in [detected_time, detected_amount] + class_cols]
                            
                            for i, col in enumerate(feature_cols):  # Use all available features
                                final_df[f'V{i+1}'] = df[col]
                        
                        # Add amount column
                        final_df['Amount'] = df[detected_amount]
                        
                        return final_df
                        
                    except Exception as e:
                        st.error(f"❌ Error preparing data: {e}")
                        return None
                
                # Try to prepare the data
                processed_df = detect_and_prepare_data(df)
                
                if processed_df is None:
                    st.error("❌ Could not process your data format. Please check your CSV file.")
                    st.markdown("""
                    **What we need:**
                    - At least one column for transaction timing
                    - At least one column for transaction amounts  
                    - Additional numeric columns for security features
                    - All values should be numeric (no text except headers)
                    """)
                    return
                
                # Update df to the processed version
                df = processed_df
                final_columns = df.shape[1]
                
                if final_columns >= 3:  # Minimum: Time, Amount, at least 1 feature
                    st.success(f"✅ Great! Successfully prepared {final_columns} columns for prediction.")
                    st.info(f"📊 Features: {final_columns-2} security features + Time + Amount")
                else:
                    st.error(f"❌ Not enough columns. Need at least 3 columns (Time, Amount, + features).")
                    return
                
                # Validate feature ranges
                st.markdown("### 📊 Feature Range Validation")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    time_range = f"{df['Time'].min():.0f} - {df['Time'].max():.0f}"
                    st.metric("⏰ Time Range (seconds)", time_range)
                    if df['Time'].max() > 200000:
                        st.warning("⚠️ Very large time values detected")
                
                with col2:
                    amount_range = f"${df['Amount'].min():.2f} - ${df['Amount'].max():.2f}"
                    st.metric("💰 Amount Range", amount_range)
                    if df['Amount'].max() > 50000:
                        st.warning("⚠️ Very large transaction amounts detected")
                
                with col3:
                    v_features = [col for col in df.columns if col.startswith('V')]
                    v_range = f"{df[v_features].min().min():.2f} - {df[v_features].max().max():.2f}"
                    st.metric("🔢 V-Features Range", v_range)
                    if abs(df[v_features].min().min()) > 10 or df[v_features].max().max() > 10:
                        st.warning("⚠️ Some V-features have extreme values")
                
                # Show data preview with better formatting
                with st.expander("📊 Data Preview (First 10 Rows)", expanded=True):
                    preview_df = df.head(10)
                    
                    # Format the display for better readability
                    styled_df = preview_df.style.format({
                        'Time': '{:.0f}',
                        'Amount': '${:.2f}'
                    })
                    
                    # Format V-features
                    v_cols = [col for col in preview_df.columns if col.startswith('V')]
                    v_format = {col: '{:.6f}' for col in v_cols}
                    styled_df = styled_df.format(v_format)
                    
                    st.dataframe(styled_df, use_container_width=True)
                    
                    # Add summary stats
                    st.caption(f"📈 Showing first 10 of {len(df):,} transactions | 📊 {len(v_cols)} V-features detected")
                
                # Processing options
                st.markdown("### ⚙️ Processing Options")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    max_rows = st.number_input(
                        "Maximum rows to process:",
                        min_value=1,
                        max_value=len(df),
                        value=min(1000, len(df)),
                        help="Limit processing for faster results. Use smaller numbers for quick testing."
                    )
                
                with col2:
                    st.metric("📁 Total File Size", f"{len(df):,} rows")
                    processing_time_est = max_rows / 1000 * 2  # Rough estimate
                    st.caption(f"⏱️ Estimated processing time: ~{processing_time_est:.1f} seconds")
                
                with col3:
                    show_detailed_results = st.checkbox(
                        "Show detailed results",
                        value=True,
                        help="Display comprehensive analysis charts and statistics"
                    )
                
                if st.button("🚀 Start Fraud Analysis", type="primary", use_container_width=True):
                    # Limit rows if specified
                    if max_rows < len(df):
                        df_subset = df.head(max_rows)
                        st.info(f"🔍 Processing first {max_rows:,} rows out of {len(df):,} total")
                    else:
                        df_subset = df
                        st.info(f"🔍 Processing all {len(df):,} transactions")
                    
                    try:
                        if show_detailed_results:
                            analyze_bulk_transactions(df_subset, detector)
                        else:
                            # Quick analysis mode
                            st.info("⚡ Quick analysis mode - processing transactions...")
                            results = detector.predict_bulk(df_subset.values.tolist())
                            fraud_count = sum(1 for r in results if r['is_fraud'])
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("🔍 Transactions Analyzed", f"{len(results):,}")
                            with col2:
                                st.metric("🚨 Fraud Detected", f"{fraud_count:,}")
                            with col3:
                                st.metric("📊 Fraud Rate", f"{fraud_count/len(results)*100:.2f}%")
                            
                            st.success("✅ Quick analysis completed!")
                            
                    except Exception as e:
                        st.error(f"❌ Analysis error: {e}")
                        st.info("""
                        **Troubleshooting Tips:**
                        - Ensure all feature values are numeric
                        - Check for missing values (NaN)
                        - Verify V-features are within reasonable ranges
                        - Try with a smaller sample size first
                        """)
                        
            except Exception as e:
                st.error(f"❌ Error reading CSV: {str(e)}")
                st.markdown("""
                **Common CSV Issues:**
                - **Encoding problems**: Try saving as UTF-8
                - **Delimiter issues**: Ensure comma-separated values
                - **Missing headers**: First row should contain column names
                - **Mixed data types**: All values should be numeric (except headers)
                - **File corruption**: Try re-exporting the data
                
                **Quick Fix:**
                1. Open your CSV in Excel or similar
                2. Verify 30-31 columns with proper names
                3. Save as CSV (UTF-8) format
                4. Try uploading again
                        """)    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>💳 Credit Card Fraud Detection | Built with Streamlit & scikit-learn</p>
        <p>🧠 Artificial Neural Network (ANN) | PCA Features | Real-time Predictions</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
