from flask import Flask
app = Flask(__name__) # webserver gateway interphase (WSGI)

@app.route('/')
def index():
    return "welcome to the face recognition web app"


if __name__ == "__main__":
    app.run(debug=True)