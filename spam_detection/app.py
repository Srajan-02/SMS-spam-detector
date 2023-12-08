from flask import Flask, render_template, request
import pickle
        
model = 'model.pkl'
classifier = pickle.load(open(model, 'rb'))
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = tfidf.transform(data).toarray()
        my_prediction = classifier.predict(vect)
        return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(debug=True)
