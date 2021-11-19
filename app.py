import numpy as np
from flask_ngrok import run_with_ngrok
from flask import Flask,request,jsonify,render_template,redirect,url_for
import pickle
app = Flask(__name__)
model = pickle.load(open('house.pkl','rb'))
@app.route('/')
def home():
  return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
  input = [float(x) for x in request.form.values()]
  final_input = [np.array(input)]
  predict = model.predict(final_input)
  
  output = round(predict[0],2)

  return render_template('index.html',output = 'House Price is : $ ()'.format(predict))
@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)
if __name__ == '__main__':
    app.run(debug=True)
