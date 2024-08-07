from flask import render_template, request




def index():
    return render_template('index.html')

def app():
    return render_template('app.html')