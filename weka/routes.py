import secrets
from shutil import copyfile
from flask import (
    render_template,
    url_for,
    flash,
    session,
    redirect,
    request,
    abort,
    Flask,
    jsonify,
    render_template_string
)
from weka.preprocessing import generic_preprocessing as gp
from weka.modules import logistic as lg
from weka.modules import naive_bayes as nb
from weka.modules import linear_svc as lsvc
from weka.visualization import visualize as vis
from weka import app
import os.path
import numpy as np
import pandas as pd

save_path = "weka/uploads/"
exts = ["csv", "json", "yaml"]



@app.route("/")
@app.route("/preprocess", methods=["GET", "POST"])
def preprocess():
    if request.method == "POST":

        if request.form["Submit"] == "Upload":
            data = request.files["data"]
            ext = data.filename.split(".")[1]
            if ext in exts:
                session["ext"] = ext
                session["fname"] = data.filename
                data.save("weka/uploads/" + data.filename)
                df = gp.read_dataset("weka/uploads/" + data.filename)
                df.to_csv("weka/clean/clean.csv")
                session["haha"] = True
                flash(f"File uploaded successully", "success")
            else:
                flash(f"Upload Unsuccessful. Please try again", "danger")

        elif request.form["Submit"] == "Delete":
            try:
                df = gp.read_dataset("weka/clean/clean.csv")
                columns = gp.get_columns(df)
                for i in columns:
                    if request.form.get(i) is not None:
                        df.drop(i, axis=1);
                df.to_csv("weka/clean/clean.csv")
                flash(f"Column(s) deleted Successfully", "success")
            except:
                flash(f"Error! Upload Dataset", "danger")
                
        elif request.form["Submit"] == "Clean":
            try:
                df = gp.read_dataset("weka/clean/clean.csv")
                columns = gp.get_columns(df)
                for i in columns:
                    if request.form.get(i) is not None:
                        
                        gp.treat_missing_numeric(df,[i],how=method)
                df.to_csv("weka/clean/clean.csv")
                flash(f"Column(s) deleted Successfully", "success")
            except:
                flash(f"Error! Upload Dataset", "danger")

    if session.get("haha") is not None:
        df = gp.read_dataset("weka/clean/clean.csv")
        description = gp.get_description(df)
        columns = gp.get_columns(df)
        dim1,dim2 = gp.get_dim(df)
        return render_template(
            "preprocess.html",
            active="preprocess",
            title="Preprocess",
            filename=session["fname"],
            no_of_rows=len(df),
            no_of_cols=len(columns),
            description=description.to_html(
                classes=[
                    "table-bordered",
                    "table-striped",
                    "table-hover",
                    "thead-light",
                    "table-responsive",
                ]
            ),
            columns=columns,
        )
    else:
        return render_template(
            "preprocess.html", active="preprocess", title="Preprocess",
        )


@app.route('/clear', methods=['GET', 'POST'])
def clear():
    session.clear()
    return render_template('preprocess.html')

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
        x_col = request.form['x_col']
        y_col = request.form['y_col']

        df = vis.xy_plot(x_col, y_col)
        heights = np.array(df[x_col]).tolist()
        weights = np.array(df[y_col]).tolist()

        newlist = []
        for h, w in zip(heights, weights):
            newlist.append({'x': h, 'y': w})
        ugly_blob = str(newlist).replace('\'', '')

        columns = vis.get_columns()
        return render_template('visualize.html', cols = columns, src = 'img/pairplot.png', xy_src = 'img/fig.png', posted = 1, data=ugly_blob)

    else:
        columns = vis.get_columns()
        return render_template('visualize.html', cols = columns, src = 'img/pairplot.png', posted = 0)
