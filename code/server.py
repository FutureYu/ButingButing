import os
from io import BytesIO
from flask import Flask, request, render_template
import base64, json
from PIL import Image
from datetime import datetime
from predict import Predictor
from rpi_define import *

app = Flask(__name__)
pred = Predictor()
pred.load_model()
received_path = BUTING_PATH + r"\received"

@app.route("/imageSubmit", methods=["POST"])
def imageSubmit():
	img = BytesIO(base64.urlsafe_b64decode(request.form['image']))
	img = Image.open(img)
	name = received_path + r"\{}-{}.png".format(today(), "".join(now().split(":")))
	if not os.path.exists(received_path):
		os.makedirs(received_path)
	img.save(name)
	return pred.predict_img(name)


@app.route('/', methods=['GET', 'POST'])
def index():
	return render_template('index.html')

if __name__ == "__main__":
	app.run()




