from flask import Flask
from canary import Tokenizer
app = Flask(__name__)

@app.route('/')
def hello_world():
    return Tokenizer('Hello - Canary')

@app.errorhandler(404)
def error_404(error):
    return "Error 404 - Canary", 404   

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)