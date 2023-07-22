from flask import Flask, render_template, request
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle

model = load_model('my_model.h5')
file_name='model.pickle'
with open(file_name, "rb") as file:
    model1 = pickle.load(file)
#model1 = pickle.load('model.pickle')

app = Flask(__name__,template_folder='templates')

@app.route ('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    T_Max = float(request.form['Maximum Temperature'])
    T_Min = float(request.form['Minimum Temperature'])
    Humidity = float(request.form['Humidity'])
    #Speed = float(request.form['Wind Speed'])
    WindPres = float(request.form['Wind Pressure'])  # Corrected parameter name

    input_data = np.array([[T_Max, T_Min, Humidity, WindPres]])
    scaler=MinMaxScaler()
    scaled=scaler.fit_transform(input_data)
    prediction1 = model.predict(scaled)
    prediction = scaler.inverse_transform(prediction1)


    output1 = prediction[0][0]
    output2 = prediction[0][1]
    output3 = prediction[0][2]
    #output4 = prediction[0][3]
    output5 = prediction[0][3]
    out = model1.predict([[output1, output2, output3, output5]])
    if out == 3:
        output6 ='Tomorrow weather is sunny'
    if out == 0:
        output6 ='Tomorrow weather is Cloudy'
    if out == 1:
        output6 ='Tomorrow weather is Rain'
    if out == 2:
        output6 ='Tomorrow weather is drizzle'


    return render_template('result.html', output1=output1, output2=output2, output3=output3, output5=output5, output6=output6)  # Render result.html page with output values

if __name__ == '__main__':
    app.run(debug=True)

