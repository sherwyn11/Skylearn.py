from flask import *
from flask_cors import CORS
import os.path
import modules.logistic as lg
import modules.naive_bayes as nb
import pandas as pd

save_path = '/uploads/'
exts = ['csv', 'json', 'yaml']

app = Flask(__name__)
app.secret_key = "dnaz&sherwu"  


@app.route('/', methods=['GET'])
def test():
    return "Hello"

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        data = request.files['data']
        ext = data.filename.split('.')[1]
        if(ext in exts):
            data.save('uploads/' + data.filename)
            session['fname'] = data.filename 
            session['ext'] = ext
            return 'File saved to uploads directory!'
        else:
            return 'File type not accepted!'
    return render_template('upload.html')


@app.route('/classify', methods=['GET', 'POST'])
def classify():
    acc = 0
    if request.method == 'POST':
        classifier = int(request.form['classifier'])
        hidden_val = int(request.form['hidden'])
        if(hidden_val == 0):
            data = request.files['choiceVal']
            ext = data.filename.split('.')[1]
            if(ext in exts):
                data.save('uploads/test.' + ext)
            else:
                return 'File type not accepted!'
            choiceVal = 0
        else:
            choiceVal = int(request.form['choiceVal'])

        if (classifier == 0):
            ret_vals = lg.logisticReg(choiceVal, hidden_val)
            if (hidden_val == 0 or hidden_val == 1):
                return render_template('classifier_page.html', acc = ret_vals[0], report = [ret_vals[1].to_html()], conf_matrix = [ret_vals[2].to_html()], choice = hidden_val, classifier_used = classifier)
            elif (hidden_val == 2):
                return render_template('classifier_page.html', acc = ret_vals[0], report = ret_vals[1], conf_matrix = ret_vals[2], choice = hidden_val, classifier_used = classifier)

        else:
            ret_vals = nb.naiveBayes(choiceVal, hidden_val)
            if (hidden_val == 0 or hidden_val == 1):
                return render_template('classifier_page.html', acc = ret_vals[0], report = [ret_vals[1].to_html()], conf_matrix = [ret_vals[2].to_html()], choice = hidden_val, classifier_used = classifier)
            elif (hidden_val == 2):
                return render_template('classifier_page.html', acc = ret_vals[0], report = ret_vals[1], conf_matrix = ret_vals[2], choice = hidden_val, classifier_used = classifier)


    elif request.method == 'GET':
        return render_template('classifier_page.html')


if __name__ == "__main__":
    app.run(debug=True)
