# 🛡️ Phishing Email Detection Model

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

A machine learning model built with **Scikit-learn** that classifies emails as **Phishing** or **Safe** based on textual content and URL features.

---

## 📁 Project Structure

```
phishing-email-detection/
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
├── main.py                     # Main training & evaluation script
├── predict.py                  # Predict on new emails
├── data/
│   └── generate_dataset.py     # Synthetic dataset generator
├── src/
│   ├── __init__.py
│   ├── feature_extraction.py   # Feature engineering pipeline
│   ├── model.py                # Model training & evaluation
│   └── utils.py                # Utility functions
└── outputs/                    # Saved models & results
    └── .gitkeep
```

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/phishing-email-detection.git
cd phishing-email-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model
```bash
python main.py
```

### 4. Predict on new emails
```bash
python predict.py
```

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **URL Analysis** | Counts URLs, detects suspicious patterns (IP addresses, shortened URLs, excessive subdomains) |
| **Keyword Detection** | Identifies phishing-related keywords (urgent, verify, suspended, etc.) |
| **Text Statistics** | Analyzes capital letter ratio, special characters, exclamation marks |
| **Urgency Scoring** | Measures urgency-related language patterns |
| **TF-IDF Vectorization** | Captures textual patterns using term frequency–inverse document frequency |
| **Ensemble Model** | Random Forest classifier with optimized hyperparameters |

---

## 📊 Model Evaluation

The model outputs:
- **Accuracy Score** — Overall classification accuracy
- **Confusion Matrix** — Visual heatmap of true vs predicted labels
- **Classification Report** — Precision, recall, F1-score per class
- **Feature Importance** — Top contributing features chart

All evaluation visuals are saved to the `outputs/` directory.

---

## 🔍 Extracted Features

The feature extraction pipeline analyzes each email for:

1. **URL Features**
   - Total URL count
   - Suspicious URL ratio (IPs, shorteners, long URLs)
   - Presence of `https` vs `http`

2. **Textual Features**
   - Phishing keyword count
   - Capital letter ratio
   - Special character count
   - Exclamation mark count
   - Email body length

3. **Behavioral Features**
   - Urgency score (deadline/action-required language)
   - HTML tag presence
   - Attachment mention detection

---

## 🧪 Dataset

This project includes a **synthetic dataset generator** (`data/generate_dataset.py`) that creates realistic phishing and legitimate email samples for training. You can also replace it with your own labeled dataset.

To generate the dataset:
```bash
python data/generate_dataset.py
```

---

## 🛠️ Tech Stack

- **Python 3.8+**
- **Scikit-learn** — Model training & evaluation
- **Pandas** — Data manipulation
- **NumPy** — Numerical operations
- **Matplotlib & Seaborn** — Visualization
- **Joblib** — Model serialization

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
