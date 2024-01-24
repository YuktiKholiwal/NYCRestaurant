# app.py
from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

# Load your pre-trained models
with open('model.pkl', 'rb') as file:
    linear_model = pickle.load(file)

with open('logistic_model.pkl', 'rb') as file:
    logistic_model = pickle.load(file)

with open('svm_model.pkl', 'rb') as file:
    svm_model = pickle.load(file)

with open('kmeans_model.pkl', 'rb') as file:
    kmeans_model = pickle.load(file)

with open('knn_model.pkl', 'rb') as file:
    knn_model = pickle.load(file)

with open('rf_model.pkl', 'rb') as file:
    rf_model = pickle.load(file)

with open('nb_model.pkl', 'rb') as file:
    nb_model = pickle.load(file)

# Repeat for other models...

@app.route('/')
def home():
    return render_template('index.html')




@app.route('/eda', methods=['POST'])
def eda():
    # Get the uploaded file
    file = request.files['file']
    
    # Read the dataset
    df = pd.read_csv(file)

    # Perform EDA (Add your EDA code here)
    # For example, let's create a bar chart of the top 10 cuisines
    selected_eda = request.form.get('eda_option')

        # Perform EDA based on the selected option
    if selected_eda == 'grade_distribution':
        grade_distribution = df.groupby(['Cities', 'Grade']).size().unstack()
        plt.figure(figsize=(8,6))
        grade_distribution.plot(kind='bar', stacked=True)
    if selected_eda == 'score':
        df['Score'].hist(bins=30)
        plt.title('Distribution of Scores')
        plt.xlabel('Score')
        plt.ylabel('Number of Establishments')
        plt.show()
    if selected_eda == 'cuisine_avg':
        avg_scores_by_cuisine = df.groupby('Cuisine Description')['Score'].mean().sort_values()
        plt.figure(figsize=(8,6))
        avg_scores_by_cuisine.plot(kind='barh')
        plt.title('Average Score by Cuisine')
        plt.xlabel('Average Score')
        plt.ylabel('Cuisine Description')
        plt.grid(axis='x')
        plt.show()
    if selected_eda == 'grade_score':
        grades = df['Grade'].unique()
        score_data = [df[df['Grade'] == grade]['Score'].values for grade in grades]
        plt.figure(figsize=(8,6))
        plt.boxplot(score_data, labels=grades)
        plt.title('Boxplot of Scores by Grade')
        plt.xlabel('Grade')
        plt.ylabel('Score')
        plt.show()
    if selected_eda == 'cf_vs_score':
        critical_flags = df['Critical Flag'].unique()
        score_data_by_flag = [df[df['Critical Flag'] == flag]['Score'].values for flag in critical_flags]
        #plotting box plot
        plt.figure(figsize=(8,6))
        plt.boxplot(score_data_by_flag, labels=critical_flags)
        plt.title('Comparison of Scores based on Critical Flag')
        plt.xlabel('Critical Flag')
        plt.ylabel('Score')
        plt.show()
    if selected_eda == 'critical_vs_noncritical':
        critical_counts = df['Critical Flag'].value_counts()
        labels = critical_counts.index
        sizes = critical_counts.values
        colors = ['red', 'green']  # Assuming "Yes" is for critical and comes first. Adjust as needed.
        explode = (0.1, 0)  # explode 1st slice for emphasis
        plt.figure(figsize=(8,6))
        plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.title("Proportion of Critical vs. Non-Critical Violations")
        plt.show()
    if selected_eda == 'vio_vs_grade':
        violation_grade = df.groupby(['Violation Code', 'Grade']).size().unstack()
        violation_grade.plot(kind='bar', stacked=True)
    if selected_eda == 'grade_vs_criticalflag':
        critical_grade_distribution = df.groupby(['Critical Flag', 'Grade']).size().unstack()
        critical_grade_distribution.plot(kind='bar', stacked=True)
        

    
    # Save the plot to a BytesIO object
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    # Convert the plot to a base64-encoded string
    plot_url = base64.b64encode(img.getvalue()).decode()

    return render_template('eda_result.html', plot_url=plot_url)


@app.route('/predict', methods=['POST'])
def predict():
    # Extract input data from the form
    data = request.form.to_dict()

    # Select the model based on the user's choice
    selected_model = data.get('model_selection', '')

    # Validate and convert input features based on the selected model
    try:
        if selected_model == 'linear':
            critical_flag = int(data.get('critical_flag', 0))
            input_data = [[critical_flag]]
            prediction = linear_model.predict(input_data)

        elif selected_model == 'logistic':
            violation_code = int(data.get('violation_code', 0))
            cuisine_description = int(data.get('cuisine_description', 0))
            input_data_logistic = [[violation_code, cuisine_description]]
            prediction = logistic_model.predict(input_data_logistic)

        
        elif selected_model == 'knn':
            violation_code = int(data.get('violation_code', 0))
            cuisine_description = int(data.get('cuisine_description', 0))
            critical_flag = int(data.get('critical_flag', 0))
            input_data_knn = [[cuisine_description, violation_code, critical_flag]]
            prediction = knn_model.predict(input_data_knn)
        
        
        elif selected_model == 'random_forest':
            score = int(data.get('score', 0))
            violation_code = int(data.get('violation_code', 0))
            cuisine_description = int(data.get('cuisine_description', 0))
            input_data_rf = [[score, violation_code, cuisine_description]]
            prediction = rf_model.predict(input_data_rf)

        elif selected_model == 'svm':
            violation_code = int(data.get('violation_code', 0))
            critical_flag = int(data.get('critical_flag', 0))
            input_data_svm = [[violation_code, critical_flag]]
            prediction = svm_model.predict(input_data_svm)

        else:
            return render_template('result.html', prediction="Invalid model selection", selected_model=selected_model)

    except ValueError:
        return render_template('result.html', prediction="Invalid input data", selected_model=selected_model)

    return render_template('result.html', prediction=prediction[0], selected_model=selected_model)

if __name__ == '__main__':
    app.run(debug=True)
