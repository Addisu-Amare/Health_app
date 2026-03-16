# Health_app

# 🏥 AI Powered multifunctional Healthcare  Hub

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-green.svg)](https://fastapi.tiangolo.com)
[![Hugging Face](https://img.shields.io/badge/🤗-HuggingFace-yellow.svg)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-MIT-red.svg)](LICENSE)

An end-to-end AI ecosystem that combines multiple machine learning models into a unified healthcare platform. Users input symptoms or queries → system delivers **diagnoses, precautions, medication alternatives, risk scores, and instant medical answers**.

---

## ✨ Features

### 🔍 Disease Prediction & Medical Recommendation
- Predicts diseases from symptoms using **RandomForest Classifier**
- Provides: medical description, precautions, medication suggestions, diet recommendations

### 💊 AI-Powered Drug Recommendation
- Finds alternative medicines using **NLP + Cosine Similarity**
- Matches based on ingredients and properties

### ❤️ Heart Disease Risk Assessment
- **LightGBM** & **EasyEnsemble** classifiers
- Inputs: age, BMI, smoking, medical history
- Output: personalized risk score + recommendations

### 🤖 MediBot - AI Health Assistant
- **Mistral-7B-Instruct** LLM via Hugging Face
- **RAG (Retrieval-Augmented Generation)** with **FAISS** vector database
- Fact-based, fast, relevant medical answers

---

## 🛠️ Tech Stack

| Component | Technologies |
|-----------|--------------|
| **ML Models** | RandomForest, LightGBM, EasyEnsemble, Scikit-learn |
| **NLP** | TF-IDF, Cosine Similarity, Transformers |
| **LLM & RAG** | Mistral-7B, Hugging Face, FAISS |
| **Backend** | Python, FastAPI, Pandas, NumPy |
| **Deployment** | Docker, CI/CD (optional) |

---

## 📁 Project Structure

```
Health_app/
│   ├── models/                       # Trained ML models
│   │   ├── first_feature_model.pkl   
│   │   ├── second_feature_model.pkl   
│   │   └── third_feature_model.pkl   
│   ├── medi_bot/                      # LLM chatbot implementation
│   │   ├── connect_memory.py          # Base memory connections
│   │   └── connect_memory_llm.py      # LLM-specific memory handling
│   └── utils/                          # Helper functions
│       └── heart_disease.jpg           # Heart visualization asset
├── data/
│   ├── datasets/
│   │  
│   └── vector_store/
│       ├── deb.faiss                   # FAISS database binary
│       └── index.faiss                  # FAISS index file
├── requirements.txt
├── Dockerfile
└── README.md
```

## 🚀 Quick Start (Conda)

```bash
# Clone repository
git clone https://github.com/Addisu-Amare/Health_app.git
cd Health_app

# Create conda environment with Python 3.9
conda create -n health_app python=3.9 -y

# Activate environment
conda activate health_app

# Install dependencies
pip install -r requirements.txt
 then   run main.py
```

---

## 🔌 API Endpoints

| Endpoint | Method | Input | Output |
|----------|--------|-------|--------|
| `/predict/disease` | POST | Symptoms list | Disease + precautions + diet |
| `/recommend/drug` | POST | Drug name | Alternative medicines |
| `/assess/heart-risk` | POST | Age, BMI, smoking, history | Risk score + recommendations |
| `/chat/medical` | POST | User query | AI medical response |

---

## 🧠 Model Details

### First Feature Model: Disease Prediction
- **Algorithm:** RandomForest Classifier
- **Features:** 132 symptoms (one-hot encoded)
-


### Second Feature Model: Drug Recommendation
- **Algorithm:** NLP + Cosine Similarity
- **Features:** TF-IDF vectors of drug ingredients



### Third Feature Model: Heart Risk Assessment
- **Algorithm:** LightGBM + EasyEnsemble
- **Features:** Age, BMI, smoking, BP, cholesterol, etc.
)


---

## 🤖 MediBot Architecture

```
User Query
    ↓
[connect_memory.py]  ←  FAISS Vector Store (deb.faiss + index.faiss)
    ↓
[connect_memory_llm.py]  ←  Mistral-7B-Instruct (Hugging Face)
    ↓
RAG-enhanced response
    ↓
User
```

---

## 🌍 Why This Matters

This system is designed for **healthcare accessibility** — especially in regions with limited medical infrastructure. It's:
- ✅ **Deployable** — not just a demo
- ✅ **Evidence-driven** — RAG ensures factual responses
- ✅ **Scalable** — API-first architecture
- ✅ **Explainable** — every recommendation comes with context

---

## 🤝 Contributing

Contributions welcome! Open an issue or submit a PR.

---

## 📄 License

MIT © Addisu Amare

---

## 📬 Contact

**Addisu Amare**  
[GitHub](https://github.com/Addisu-Amare) | [LinkedIn](www.linkedin.com/in/addisu-amare-2643ba16a) | [Email](mailto:0941813057estifanos@gmail.com)

---

> *Making healthcare smarter, faster, and accessible — one query at a time.*
