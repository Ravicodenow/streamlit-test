# 💳 Credit Card Fraud Detection - Single File Streamlit App

A complete, self-contained credit card fraud detection application built with **Streamlit** and **scikit-learn**. This single-file application includes machine learning model training, real-time predictions, and interactive data visualization.

## 🌟 Features

### 🧠 Machine Learning
- **Artificial Neural Network (ANN)** with 3 hidden layers (64→32→16 neurons)
- **Real-time training** with synthetic or uploaded datasets
- **Balanced class weights** to handle imbalanced fraud data
- **ROC-AUC evaluation** and performance metrics

### 🎯 Prediction Methods
- **Manual Input**: Enter 30 transaction features manually
- **Sample Generator**: Generate and analyze random transactions
- **Bulk Analysis**: Upload CSV files for batch processing

### 📊 Visualizations
- **Interactive gauges** for fraud probability
- **Risk distribution charts** (High/Medium/Low risk)
- **Fraud probability histograms**
- **Real-time results dashboard**

### 🔧 Additional Features
- **CSV export** of analysis results
- **Progress tracking** for bulk operations
- **Responsive design** for different screen sizes
- **Error handling** and user guidance

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r streamlit_requirements.txt
```

### 2. Run the Application
```bash
# Option 1: Use the startup script
./run_streamlit_app.sh

# Option 2: Run directly with streamlit
streamlit run streamlit_fraud_detector.py
```

### 3. Access the App
Open your browser and navigate to: **http://localhost:8501**

## 📋 How to Use

### Step 1: Train the Model
1. Open the **sidebar** on the left
2. Choose training data:
   - **Sample Dataset**: Uses synthetic fraud data (perfect for demo)
   - **Upload CSV**: Use your own dataset with fraud labels
3. Click **"🚀 Train Model"**
4. Wait for training to complete (~30 seconds)

### Step 2: Analyze Transactions
Choose one of three analysis methods:

#### 🔢 Manual Input
- Enter all 30 transaction features manually
- Perfect for testing specific transaction patterns
- Get instant fraud probability and risk assessment

#### 🎲 Sample Transaction
- Generate random realistic transactions
- One-click analysis for quick demonstrations
- See how the model responds to different scenarios

#### 📄 Bulk CSV Upload
- Upload CSV files with multiple transactions
- Batch processing for thousands of transactions
- Comprehensive analysis dashboard with visualizations
- Export results as CSV

## 📊 Data Format

The model expects **30 features** in this exact order:

| Feature | Description | Example |
|---------|-------------|---------|
| Time | Seconds since first transaction | 0, 172800 |
| V1-V28 | PCA-transformed features | -1.359, 1.191, ... |
| Amount | Transaction amount in dollars | 149.62 |

For training data, add a **31st column**:
| Class | Target variable | 0 (Legitimate), 1 (Fraud) |

### Sample CSV Format
```csv
Time,V1,V2,V3,...,V28,Amount,Class
0,-1.359807134,-0.072781173,2.536346738,...,-0.021053053,149.62,0
1,1.191857,0.266151,0.166480,...,0.014724,2.69,0
```

## 🔧 Technical Details

### Model Architecture
- **Input Layer**: 30 features
- **Hidden Layer 1**: 64 neurons (ReLU activation)
- **Hidden Layer 2**: 32 neurons (ReLU activation)  
- **Hidden Layer 3**: 16 neurons (ReLU activation)
- **Output Layer**: 1 neuron (Sigmoid activation)
- **Regularization**: L2 (α=0.001) + Early Stopping

### Performance Features
- **Batch Processing**: Handles large datasets efficiently
- **Memory Management**: Processes data in chunks
- **Progress Tracking**: Real-time progress bars
- **Error Handling**: Graceful failure recovery

### Dependencies
- `streamlit>=1.28.0` - Web application framework
- `pandas>=2.0.0` - Data manipulation
- `numpy>=1.24.0` - Numerical computing
- `plotly>=5.17.0` - Interactive visualizations
- `scikit-learn>=1.3.0` - Machine learning
- `joblib>=1.3.0` - Model serialization

## 📈 Use Cases

### 🏦 Financial Institutions
- Real-time transaction monitoring
- Batch processing of daily transactions
- Risk assessment dashboards
- Compliance reporting

### 🎓 Educational & Research
- Machine learning demonstrations
- Fraud detection algorithm testing
- Data science training materials
- Academic research projects

### 💼 Business Analytics
- Transaction pattern analysis
- Customer behavior insights
- Risk management tools
- Data-driven decision making

## 🛠️ Customization

### Modify Model Parameters
Edit the `build_ann_model()` function in `streamlit_fraud_detector.py`:
```python
model = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),  # Change architecture
    alpha=0.01,                        # Adjust regularization
    max_iter=1000,                     # More training epochs
    # ... other parameters
)
```

### Adjust UI Layout
Modify the Streamlit layout in the `main()` function:
- Change column layouts: `st.columns([2, 1, 3])`
- Add new input widgets: `st.slider()`, `st.selectbox()`
- Customize CSS styles in the `st.markdown()` sections

### Add New Features
- **Model Comparison**: Train multiple algorithms
- **Feature Engineering**: Add new derived features
- **Advanced Visualizations**: Create custom plots
- **Export Options**: Add PDF/Excel export

## ⚡ Performance Tips

### For Large Datasets
1. **Limit Processing**: Use the "Maximum rows" setting
2. **Batch Size**: Process in smaller chunks if memory is limited
3. **Sample Data**: Train on a representative subset first

### For Better Accuracy
1. **More Training Data**: Use larger, diverse datasets
2. **Feature Engineering**: Add domain-specific features
3. **Hyperparameter Tuning**: Experiment with model parameters
4. **Cross-Validation**: Implement k-fold validation

## 🔒 Security Notes

- **Data Privacy**: All processing happens locally
- **No Data Transmission**: No external API calls
- **Secure Upload**: Files are processed in memory only
- **Session Isolation**: Each user session is independent

## 📞 Support & Troubleshooting

### Common Issues

**Model Won't Train**
- Check data format (30-31 columns)
- Ensure numeric values only
- Verify CSV file encoding

**Slow Performance**
- Reduce batch size for bulk processing
- Limit number of rows processed
- Close other applications to free memory

**Memory Errors**
- Process smaller chunks of data
- Restart the application
- Increase available system memory

### Getting Help
1. Check the browser console for error messages
2. Verify all dependencies are installed correctly
3. Try with the sample dataset first
4. Restart the Streamlit application

## 🏆 About

This application demonstrates the power of combining **machine learning** with **interactive web interfaces**. Built entirely in Python, it showcases:

- **End-to-end ML pipeline**: From data loading to prediction
- **Modern web UI**: Responsive, intuitive interface
- **Real-time processing**: Instant feedback and results
- **Production-ready code**: Error handling, validation, optimization

Perfect for **data scientists**, **financial analysts**, **researchers**, and anyone interested in **fraud detection** and **machine learning applications**.

---

**Built with ❤️ using Streamlit, scikit-learn, and Python**