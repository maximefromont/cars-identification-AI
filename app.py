
# Import de bibliothèques
import flask
from flask import request, jsonify
import os
data, isSorted = False, False


if not os.path.exists('the-car-connection-picture-dataset'):
    os.system('python get-dataset-from-kaggle.py')
data = True
# Si dossier sorted data n'existe pas, executer, sort_data_into_folder.py
if not os.path.exists('sorted_data'):
    os.system('python sort_data_into_folder.py')
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
   return jsonify({result: "Model", probability: 0.99})
   

app.run()