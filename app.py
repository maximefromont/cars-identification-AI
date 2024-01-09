
# Import de bibliothèques
import flask
from flask import request, jsonify
import os
from datasetdwl import get_dataset_from_kaggle, sort_dataset_into_folder


data, isSorted = False, False
if not os.path.exists('the-car-connection-picture-dataset'):
    get_dataset_from_kaggle.main()
data = True
# Si dossier sorted data n'existe pas, executer, sort_data_into_folder.py
if not os.path.exists('sorted-dataset'):
    print("Sorting data into folder...")
    #sort_dataset_into_folder
isSorted = True


# Création de l'objet Flask
app = flask.Flask(__name__)

# Lancement du Débogueur
app.config["DEBUG"] = True
app.config['JSON_AS_ASCII'] = False


@app.route('/', methods=['GET'])
def home():
    return [{"data": data, "isSorted": isSorted}]


@app.route('/input', methods=['POST'])
def input():
    photo = request.files['photo']
    return jsonify({'result': "Model", 'probability': 0.99})
   

#app.run()