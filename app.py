from flask import Flask, render_template, request, jsonify
import requests
import json

def get_api(method, **params):
	url = "http://ec2-54-173-246-152.compute-1.amazonaws.com:7777/"
	payload = {
	    "jsonrpc": "2.0",
	    "method": method,
	    "params": params,
	    "id": 1
	}
	headers = {"Content-Type": "application/json"}
	response = requests.request("GET", url, json=payload, headers=headers)
	return json.loads(response.text)['result']

print(get_api('alive'))

app = Flask(__name__)

@app.route("/")
@app.route("/index")
def index():
	return render_template("index.html")

@app.route('/predict', methods = ['POST'])
def predict():
	data = request.get_json()
	print(data)
	return jsonify(get_api('predict', **data))

if __name__ == '__main__':
	app.run(debug=True)