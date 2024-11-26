from flask import Flask
from flask import request
from flask import jsonify

from pickle_utils import load_from_pickle

dv = load_from_pickle('./model/dv.pkl')
model = load_from_pickle('./model/model.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    potability = y_pred >= 0.5

    result = {
        'probability': float(y_pred),
        'potability': bool(potability)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
