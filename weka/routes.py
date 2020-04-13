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
                df = gp.delete_column(df, request.form.getlist("check_cols"))
                df.to_csv("weka/clean/clean.csv", mode="w", index=False)
                flash(f"Column(s) deleted Successfully", "success")
            except:
                flash(f"Error! Upload Dataset", "danger")

        elif request.form["Submit"] == "Clean":
            try:
                df = gp.read_dataset("weka/clean/clean.csv")
                print(request.form["how"])
                if request.form["how"] is not "any":
                    df = gp.treat_missing_numeric(
                        df, request.form.getlist("check_cols"), how=request.form["how"]
                    )
                elif request.form["howNos"] is not None:
                    df = gp.treat_missing_numeric(
                        df,
                        request.form.getlist("check_cols"),
                        how=float(request.form["howNos"]),
                    )

                df.to_csv("weka/clean/clean.csv", mode="w", index=False)
                flash(f"Column(s) cleant Successfully", "success")
            except:
                flash(f"Error! Upload Dataset", "danger")

    if session.get("haha") is not None:
        df = gp.read_dataset("weka/clean/clean.csv")
        description = gp.get_description(df)
        columns = gp.get_columns(df)
        print(columns)
        dim1, dim2 = gp.get_dim(df)
        head = gp.get_head(df)
        return render_template(
            "preprocess.html",
            active="preprocess",
            title="Preprocess",
            filename=session["fname"],
            no_of_rows=len(df),
            no_of_cols=len(columns),
            dim=str(dim1) + " x " + str(dim2),
            description=description.to_html(
                classes=[
                    "table-bordered",
                    "table-striped",
                    "table-hover",
                    "thead-light",
                ]
            ),
            columns=columns,
            head=head.to_html(
                classes=[
                    "table",
                    "table-bordered",
                    "table-striped",
                    "table-hover",
                    "thead-light",
                ]
            ),
        )
    else:
        return render_template(
            "preprocess.html", active="preprocess", title="Preprocess",
        )


@app.route("/classify", methods=["GET", "POST"])
def classify():
    acc = 0
    if request.method == "POST":
        classifier = int(request.form["classifier"])
        hidden_val = int(request.form["hidden"])
        scale_val = int(request.form["scale_hidden"])
        encode_val = int(request.form["encode_hidden"])

        if hidden_val == 0:
            data = request.files["choiceVal"]
            ext = data.filename.split(".")[1]
            if ext in exts:
                data.save("uploads/test." + ext)
            else:
                return "File type not accepted!"
            choiceVal = 0
        else:
            choiceVal = int(request.form["choiceVal"])

        if classifier == 0:
            ret_vals = lg.logisticReg(choiceVal, hidden_val, scale_val, encode_val)
            if hidden_val == 0 or hidden_val == 1:
                return render_template(
                    "classifier_page.html",
                    acc=ret_vals[0],
                    report=[
                        ret_vals[1].to_html(
                            classes=[
                                "table",
                                "table-bordered",
                                "table-striped",
                                "table-hover",
                                "thead-light",
                            ]
                        )
                    ],
                    conf_matrix=[
                        ret_vals[2].to_html(
                            classes=[
                                "table",
                                "table-bordered",
                                "table-striped",
                                "table-hover",
                                "thead-light",
                            ]
                        )
                    ],
                    choice=hidden_val,
                    classifier_used=classifier,
                    active="classify",
                    title="Classify"
                )
            elif hidden_val == 2:
                return render_template(
                    "classifier_page.html",
                    acc=ret_vals[0],
                    report=ret_vals[1],
                    conf_matrix=ret_vals[2],
                    choice=hidden_val,
                    classifier_used=classifier,
                    active="classify",
                    title="Classify"
                )

        elif classifier == 1:
            ret_vals = nb.naiveBayes(choiceVal, hidden_val, scale_val, encode_val)
            if hidden_val == 0 or hidden_val == 1:
                return render_template(
                    "classifier_page.html",
                    acc=ret_vals[0],
                    report=[
                        ret_vals[1].to_html(
                            classes=[
                                "table",
                                "table-bordered",
                                "table-striped",
                                "table-hover",
                                "thead-light",
                            ]
                        )
                    ],
                    conf_matrix=[
                        ret_vals[2].to_html(
                            classes=[
                                "table",
                                "table-bordered",
                                "table-striped",
                                "table-hover",
                                "thead-light",
                            ]
                        )
                    ],
                    choice=hidden_val,
                    classifier_used=classifier,
                    active="classify",
                    title="Classify"
                )
            elif hidden_val == 2:
                return render_template(
                    "classifier_page.html",
                    acc=ret_vals[0],
                    report=ret_vals[1],
                    conf_matrix=ret_vals[2],
                    choice=hidden_val,
                    classifier_used=classifier,
                    active="classify",
                    title="Classify"
                )

        elif classifier == 2:
            ret_vals = lsvc.lin_svc(choiceVal, hidden_val, scale_val, encode_val)
            if hidden_val == 0 or hidden_val == 1:
                return render_template(
                    "classifier_page.html",
                    acc=ret_vals[0],
                    report=[
                        ret_vals[1].to_html(
                            classes=[
                                "table",
                                "table-bordered",
                                "table-striped",
                                "table-hover",
                                "thead-light",
                            ]
                        )
                    ],
                    conf_matrix=[
                        ret_vals[2].to_html(
                            classes=[
                                "table",
                                "table-bordered",
                                "table-striped",
                                "table-hover",
                                "thead-light",
                            ]
                        )
                    ],
                    choice=hidden_val,
                    classifier_used=classifier,
                    active="classify",
                    title="Classify"
                )
            elif hidden_val == 2:
                return render_template(
                    "classifier_page.html",
                    acc=ret_vals[0],
                    report=ret_vals[1],
                    conf_matrix=ret_vals[2],
                    choice=hidden_val,
                    classifier_used=classifier,
                    active="classify",
                    title="Classify"
                )

    elif request.method == "GET":
        return render_template("classifier_page.html", active="classify", title="Classify")


@app.route("/clear", methods=["GET"])
def clear():
    session.clear()
    return redirect("/")


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
        print(x_col)
        return render_template(
            'visualize.html', 
            cols = columns, 
            src = 'img/pairplot.png', 
            xy_src = 'img/fig.png', 
            posted = 1, 
            data=ugly_blob, 
            active = 'visualize',
            x_col_name = str(x_col),
            y_col_name = str(y_col),
            title="Visualize"
        )

    else:
        vis.pair_plot()
        columns = vis.get_columns()
        return render_template(
            'visualize.html', 
            cols = columns, 
            src = 'img/pairplot.png', 
            posted = 0, 
            active = 'visualize',
            title="Visualize"
        )