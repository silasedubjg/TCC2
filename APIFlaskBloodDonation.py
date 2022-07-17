from flask import Flask, jsonify, request
from flask_restful import Resource, Api
from joblib import load
import numpy as np

# from flask import Flask
# app = Flask(__name__)

# @app.route('/')
# def hello_world():
#     return 'Hello, World!'

app = Flask(__name__)

api = Api(app)

model = load('C:/Users/silas/ET.joblib')

class BloodDonation(Resource):
    def get(self):
        return {'Trabalho de Conclusao de Curso': 'Classificacao de doadores de sangue'}
    def post(self):
        args = request.get_json(force=True)
        input_values = np.asarray(list(args.values())).reshape(1, -1)
        predict = model.predict(input_values)[0]

        return jsonify({'classificador': float(predict)})

api.add_resource(BloodDonation, '/')

if __name__ == '__main__':
    app.run()