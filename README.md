🤖 Question Generation Model

An end-to-end machine learning pipeline for **automatic question generation** — from raw data preparation to model training and deployment via a web app.

📁 Project Structure

```
question-generation/
│
├── data/                        # Raw and processed datasets
│
├── model/
│   └── checkpoint-500/          # Saved model checkpoint (step 500)
│
├── question_model/              # Final trained model files
│
├── prepare_data.py              # Data preprocessing and formatting
├── generate_questions.py        # Question generation inference script
├── train_model.py               # Model training script
├── app.py                       # Web application (API / UI)
└── README.md
```

🚀 Getting Started

Prerequisites

- Python 3.8+
- pip

### Installation

```bash
git clone https://github.com/codewithaditya47/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

---

## 🔄 Pipeline

### 1. Prepare Data

```bash
python prepare_data.py
```

Preprocesses raw input data and formats it for model training. Output is saved to the `data/` directory.
2. Train the Model

```bash
python train_model.py
```

Fine-tunes the model on the prepared dataset. Checkpoints are saved under `model/checkpoint-500/`.

### 3. Generate Questions

```bash
python generate_questions.py
```

Runs inference using the trained model from `question_model/` to generate questions from input text.

### 4. Run the App

```bash
python app.py
```

Launches the web application for interacting with the model.

---

## 📊 Model

| Detail | Value |
|--------|-------|
| Task | Question Generation |
| Checkpoint | `model/checkpoint-500` |
| Final Model | `question_model/` |

---
Made with by [codewithaditya47](https://github.com/codewithaditya47)

