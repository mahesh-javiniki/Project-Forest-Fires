from flask import Flask, render_template, request, jsonify
import pickle

application = Flask(__name__)
app=application

model = pickle.load(open('models/regressor.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))


@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predict-data', methods=['GET', 'POST'])
def predict_data():
    if request.method=="POST":
        day = int(request.form.get('day'))
        month = int(request.form.get('month'))
        Temperature=float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        DC = float(request.form.get('DC'))
        ISI = float(request.form.get('ISI'))
        BUI = float(request.form.get('BUI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_data_scaled=scaler.transform(
            [[day,month,Temperature,RH,Ws,Rain,FFMC,DMC,DC,ISI,BUI,Classes,Region]]
        )
        result=model.predict(new_data_scaled)

        return render_template('home.html',results=result[0])
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0")

