from flask import Flask, render_template, request
import joblib, os, json, numpy as np

app = Flask(__name__)

BASE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE, '..', 'best_model_final.joblib')
SCALER_PATH = os.path.join(BASE, '..', 'best_scaler_final.joblib')
COLS_PATH = os.path.join(BASE, '..', 'model_columns_final.json')

if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(BASE, 'best_model_final.joblib')
if not os.path.exists(SCALER_PATH):
    SCALER_PATH = os.path.join(BASE, 'best_scaler_final.joblib')
if os.path.exists(COLS_PATH):
    with open(COLS_PATH,'r') as f:
        MODEL_COLUMNS = json.load(f)
else:
    MODEL_COLUMNS = ['systolic_num','diastolic_num']

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

def map_form_to_vector(form):
    fv = {c:0 for c in MODEL_COLUMNS}
    try:
        fv['systolic_num'] = float(form.get('systolic','120'))
        fv['diastolic_num'] = float(form.get('diastolic','80'))
    except:
        fv['systolic_num'] = 120.0
        fv['diastolic_num'] = 80.0
    for key in ['History','TakeMedication','BreathShortness','VisualChanges','NoseBleeding','ControlledDiet']:
        if key in MODEL_COLUMNS:
            fv[key] = 1 if form.get(key,'no').lower()=='yes' else 0
        else:
            if key.lower() in MODEL_COLUMNS:
                fv[key.lower()] = 1 if form.get(key,'no').lower()=='yes' else 0
    g = form.get('gender','other').lower()
    if 'gender_female' in MODEL_COLUMNS:
        fv['gender_female'] = 1 if g=='female' else 0
    if 'gender_male' in MODEL_COLUMNS:
        fv['gender_male'] = 1 if g=='male' else 0
    age_val = form.get('age','').strip()
    if age_val:
        col = f'age_{age_val}'
        if col in MODEL_COLUMNS:
            fv[col] = 1
    sev_val = form.get('severity','').strip()
    if sev_val:
        col = f'sev_{sev_val}'
        if col in MODEL_COLUMNS:
            fv[col] = 1
    arr = np.array([fv[c] for c in MODEL_COLUMNS]).reshape(1,-1).astype(float)
    return arr

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/details')
def details():
    return render_template('details.html')

@app.route('/prediction', methods=['GET','POST'])
def prediction():
    if request.method == 'POST':
        form = request.form
        arr = map_form_to_vector(form)
        arr_scaled = scaler.transform(arr)
        pred = model.predict(arr_scaled)[0]
        try:
            proba = float(model.predict_proba(arr_scaled)[0,1])
        except:
            proba = None
        scenario = form.get('scenario','scenario1')
        advice = ''
        if scenario == 'scenario1':
            if pred == 1:
                advice = 'Alert: High BP spike predicted — notify patient & provider immediately.'
            else:
                advice = 'No immediate spike predicted.'
        elif scenario == 'scenario2':
            sys = float(form.get('systolic',120))
            if sys >= 140:
                advice = 'Fitness Advice: reduce workout intensity, rest, re-check BP.'
            elif sys >= 120:
                advice = 'Fitness Advice: moderate intensity, hydrate, monitor.'
            else:
                advice = 'Fitness Advice: OK to continue current plan.'
        elif scenario == 'scenario3':
            if pred == 1:
                advice = 'Population Health: flag patient for outreach & telehealth.'
            else:
                advice = 'No flag required.'
        return render_template('prediction.html', result=pred, proba=proba, advice=advice, form=form)
    else:
        return render_template('prediction.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
