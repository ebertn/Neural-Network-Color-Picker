import sys

from flask import Flask, request, render_template, redirect, Response
import random, json

#from python.logreg import LogReg
from python.neuralnetwork import NeuralNetwork

app = Flask(__name__)

# set the project root directory as the static folder, you can set others.
#

# Serve html to user
@app.route('/')
def output():
    return render_template('index.html')

# Receive data when the user clicks a box
@app.route('/receiver', methods = ['POST'])
def worker():
	# read json + reply
	data = request.get_json(force=True)
	result = str(data)[1:-1] + '\n'

	# Write color data to file
	with open('./data/labelled_colors.csv','a') as fd:
		fd.write(result)

	print(result)

	return result

if __name__ == "__main__":
	
	network = NeuralNetwork.fromCsv((3, 5, 2), './data/labelled_colors.csv')
	
	#app.run()