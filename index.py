from flask import Flask, render_template, request
from werkzeug import secure_filename

app = Flask(__name__)

@app.route('/')
def root():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/tutorial')
def tutorial():
    return render_template('tutorial.html')

@app.route('/input')
def input():
    return render_template('input.html')

@app.route('/application')
def application():
    return render_template('application.html')

@app.route('/upload', methods = ['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(f.filename))
        return ('file uploaded: ' + str(f.filename))

@app.errorhandler(404)
def error_404(error):
    return render_template('404.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)