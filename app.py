from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load scaler and model
scaler = pickle.load(open("Models/scaler.pkl", 'rb'))
model = pickle.load(open("Models/model.pkl", 'rb'))

class_names = [
    'Lawyer','Doctor','Government Officer','Artist','Unknown',
    'Software Engineer','Teacher','Business Owner','Scientist',
    'Banker','Writer','Accountant','Designer',
    'Construction Engineer','Game Developer','Stock Investor',
    'Real Estate Developer'
]

def Recommendation_system(
    gender, part_time_job, extracurricular_activities, absence_days,
    weekly_self_study_hours, math_score, history_score, physics_score,
    chemistry_score, biology_score, geography_score, english_score,
    total_score, average_score
):
    # Encode categorical features
    gender_encoded = 1 if gender.lower() == 'female' else 0
    part_time_job_encoded = 1 if part_time_job else 0
    extracurricular_activities_encoded = 1 if extracurricular_activities else 0

    # Create feature array
    feature_array = np.array([[gender_encoded, part_time_job_encoded, absence_days,
                               extracurricular_activities_encoded, weekly_self_study_hours,
                               math_score, history_score, physics_score, chemistry_score,
                               biology_score, geography_score, english_score,
                               total_score, average_score]])
    
    # Scale features
    scaled_features = scaler.transform(feature_array)

    # Predict probabilities
    probabilities = model.predict_proba(scaled_features)

    # Get top 5 classes
    top_classes_idx = np.argsort(-probabilities[0])[:5]
    top_classes_names_probs = [
        (class_names[idx], round(probabilities[0][idx]*100, 2)) for idx in top_classes_idx
    ]

    return top_classes_names_probs


@app.route("/")
def home():
    return render_template("home.html")


@app.route('/recommend')
def recommend():
    return render_template("recommend.html")


@app.route("/pred", methods=["POST", "GET"])
def pred():
    if request.method == "POST":
        gender = request.form['gender']

        # Convert boolean-like fields
        part_time_job = request.form.get('part_time_job', 'no').lower() in ["yes", "true", "1", "on"]
        extracurricular_activities = request.form.get('extracurricular_activities', 'no').lower() in ["yes", "true", "1", "on"]

        # Numeric fields
        absence_days = int(request.form['absence_days'])
        weekly_self_study_hours = int(request.form['weekly_self_study_hours'])
        math_score = int(request.form['math_score'])
        history_score = int(request.form['history_score'])
        physics_score = int(request.form['physics_score'])
        chemistry_score = int(request.form['chemistry_score'])
        biology_score = int(request.form['biology_score'])
        english_score = int(request.form['english_score'])
        geography_score = int(request.form['geography_score'])
        total_score = float(request.form['total_score'])
        avg_score = float(request.form['avg_score'])

        # Get recommendations
        recommendations = Recommendation_system(
            gender, part_time_job, extracurricular_activities, absence_days,
            weekly_self_study_hours, math_score, history_score, physics_score,
            chemistry_score, biology_score, geography_score, english_score,
            total_score, avg_score
        )

        return render_template('result.html', recommendations=recommendations)

    return render_template("home.html")


if __name__ == "__main__":
    app.run(debug=True)
