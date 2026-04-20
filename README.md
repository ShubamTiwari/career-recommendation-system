# 🎓 Career Recommendation System

A machine learning web application that recommends the top 5 career paths for students based on their academic scores, habits, and interests.

## 🚀 Demo

> Enter your subject scores, study habits, and profile → Get personalized career recommendations instantly.

## 🛠️ Tech Stack

- **Frontend:** HTML, CSS (Flask Templates)
- **Backend:** Python, Flask
- **ML Model:** Scikit-learn (trained on student scores dataset)
- **Data Processing:** Pandas, NumPy, SMOTE (for class balancing)

## 📁 Project Structure
career_project/
├── app.py               # Flask app & prediction logic
├── Models/
│   ├── model.pkl        # Trained ML model
│   └── scaler.pkl       # Feature scaler
├── templates/           # HTML pages
├── static/              # CSS, images
├── Model_train.ipynb    # Model training notebook
└── student-scores.csv   # Dataset
## ⚙️ How to Run Locally

1. Clone the repository
```bash
   git clone https://github.com/ShubamTiwari/career-recommendation-system.git
   cd career-recommendation-system
```

2. Install dependencies
```bash
   pip install flask numpy pandas scikit-learn imbalanced-learn
```

3. Run the app
```bash
   python app.py
```

4. Open your browser and go to `http://127.0.0.1:5000`

## 🎯 Career Categories Predicted

Lawyer, Doctor, Government Officer, Artist, Software Engineer, Teacher,
Business Owner, Scientist, Banker, Writer, Accountant, Designer,
Construction Engineer, Game Developer, Stock Investor, Real Estate Developer

## 📊 Dataset

- Source: `student-scores.csv`
- Features: Gender, subject scores (Math, Physics, Chemistry, Biology, History, Geography, English), study hours, extracurricular activities, absence days

## 🤝 Contributing

Pull requests are welcome!