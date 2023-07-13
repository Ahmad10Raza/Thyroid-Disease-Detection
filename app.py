import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import warnings
warnings.filterwarnings('ignore')

pickled_model = pickle.load(open('notebooks/random_forest_model.pkl', 'rb'))

app = Flask(__name__)



@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    

    age = float(request.form.get('age', False))
    sex = float(request.form.get('sex', False))
    TSH = float(request.form.get('TSH', False))
    T3 = float(request.form.get('T3', False))
    T4U = float(request.form.get('T4U', False))
    FTI = float(request.form.get('FTI', False))
    onthyroxine = float(request.form.get('onthyroxine', False))
    queryonthyroxine = float(request.form.get('queryonthyroxine', False))
    onantithyroidmedication = float(request.form.get('onantithyroidmedication', False))
    sick = float(request.form.get('sick', False))
    pregnant = float(request.form.get('pregnant', False))
    thyroidsurgery = float(request.form.get('thyroidsurgery', False))
    I131treatment = float(request.form.get('I131treatment', False))
    queryhypothyroid = float(request.form.get('queryhypothyroid', False))
    queryhyperthyroid = float(request.form.get('queryhyperthyroid', False))
    lithium = float(request.form.get('lithium', False))
    goitre = float(request.form.get('goitre', False))
    tumor = float(request.form.get('tumor', False))
    hypopituitary = float(request.form.get('hypopituitary', False))
    psych = float(request.form.get('psych', False))

    
    values = ({"age": age, "sex": sex,
               "TSH": TSH, "T3": T3, "T4U": T4U, "FTI": FTI,
               "onthyroxine": onthyroxine, "queryonthyroxine": queryonthyroxine,
               "onantithyroidmedication": onantithyroidmedication,
               "sick": sick, "pregnant": pregnant, "thyroidsurgery": thyroidsurgery,
               "I131treatment": I131treatment,
               "queryhypothyroid": queryhypothyroid, "queryhyperthyroid": queryhyperthyroid,
               "lithium": lithium, "goitre": goitre, "tumor": tumor,
               "hypopituitary": hypopituitary,
               "psych": psych})

    

    df_transform = pd.DataFrame.from_dict([values])

    # print("applying transformation\n")

    df_transform.age = df_transform['age'] ** (1 / 2)
    print(df_transform.age)

    df_transform.TSH = np.log1p(df_transform['TSH'])
    # print(df_transform.TSH)
    #
    df_transform.T3 = df_transform['T3'] ** (1 / 2)
    # print(df_transform.T3)

    df_transform.T4U = np.log1p(df_transform['T4U'])
    # print(df_transform.T4U)

    df_transform.FTI = df_transform['FTI'] ** (1 / 2)
    # print(df_transform.FTI)

    df_transform.to_dict()

    arr = np.array([[df_transform.age, sex,
                     df_transform.TSH, df_transform.T3, df_transform.T4U, df_transform.FTI,
                     onthyroxine, queryonthyroxine,
                     onantithyroidmedication,
                     sick, pregnant, thyroidsurgery,
                     I131treatment,
                     queryhypothyroid, queryhyperthyroid,
                     lithium, goitre, tumor,
                     hypopituitary,
                     psych]])

    # print("After transformation:\n")
    # print(arr)

    pred = pickled_model.predict(arr)[0]

    if pred == 0:
        res_Val = "Hyperthyroid"
    elif pred == 1:
        res_Val = "Hypothyroid"
    else:
        res_Val = 'Negative'

    return render_template('result.html', prediction_text='Result: {}'.format(res_Val))


if __name__ == '__main__':
    app.run(debug=True, port=5000)
