from flask import *
import os.path
import modules.logistic as lg
import modules.naive_bayes as nb
import modules.linear_svc as lsvc
import visualization.visualize as vis
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
        scale_val = int(request.form['scale_hidden'])
        encode_val = int(request.form['encode_hidden'])

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
            ret_vals = lg.logisticReg(choiceVal, hidden_val, scale_val, encode_val)
            if (hidden_val == 0 or hidden_val == 1):
                return render_template('classifier_page.html', acc = ret_vals[0], report = [ret_vals[1].to_html()], conf_matrix = [ret_vals[2].to_html()], choice = hidden_val, classifier_used = classifier)
            elif (hidden_val == 2):
                return render_template('classifier_page.html', acc = ret_vals[0], report = ret_vals[1], conf_matrix = ret_vals[2], choice = hidden_val, classifier_used = classifier)

        elif (classifier == 1):
            ret_vals = nb.naiveBayes(choiceVal, hidden_val, scale_val, encode_val)
            if (hidden_val == 0 or hidden_val == 1):
                return render_template('classifier_page.html', acc = ret_vals[0], report = [ret_vals[1].to_html()], conf_matrix = [ret_vals[2].to_html()], choice = hidden_val, classifier_used = classifier)
            elif (hidden_val == 2):
                return render_template('classifier_page.html', acc = ret_vals[0], report = ret_vals[1], conf_matrix = ret_vals[2], choice = hidden_val, classifier_used = classifier)

        elif (classifier == 2):
            ret_vals = lsvc.lin_svc(choiceVal, hidden_val, scale_val, encode_val)
            if (hidden_val == 0 or hidden_val == 1):
                return render_template('classifier_page.html', acc = ret_vals[0], report = [ret_vals[1].to_html()], conf_matrix = [ret_vals[2].to_html()], choice = hidden_val, classifier_used = classifier)
            elif (hidden_val == 2):
                return render_template('classifier_page.html', acc = ret_vals[0], report = ret_vals[1], conf_matrix = ret_vals[2], choice = hidden_val, classifier_used = classifier)


    elif request.method == 'GET':
        return render_template('classifier_page.html')

@app.route('/visualize', methods=['GET', 'POST'])
def visualize():
    if (request.method == 'POST'):
        vis.pair_plot()
        return render_template('visualize.html', posted = 1)
    else:
        return render_template('visualize.html')

if __name__ == "__main__":
    app.run(debug=True)
