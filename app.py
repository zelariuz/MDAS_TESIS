from flask import Flask, render_template, request, jsonify
import requests
import json

MODE_DEV = False

def get_api(method, **params):
	global MODE_DEV
	url = 'http://localhost:7777/' if (MODE_DEV == True) else "http://ec2-15-229-2-232.sa-east-1.compute.amazonaws.com:7777/"
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