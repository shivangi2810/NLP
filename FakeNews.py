from flask import Flask, render_template, request, jsonify
import nltk
import pickle
from nltk.corpus import stopwords
import re
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)
ps = PorterStemmer()

# Load model and count vectorizer
LR_model = pickle.load(open('FakeNewsClassifier_Logreg.pkl', 'rb'))
NB_model = pickle.load(open('FakeNewsClassifier_NB.pkl', 'rb'))
cv = pickle.load(open('FakeNewsClassifier_CV.pkl', 'rb'))
tfidf = pickle.load(open('FakeNewsClassifier_TFIDF.pkl', 'rb'))


# Build functionalities
@app.route('/', methods=['GET'])
def home():
    return render_template('index2.html')

def make_prediction(text, selectedModel, selectedVect):
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)

    if selectedVect == "Count Vectorizer":
        review_vect = cv.transform([review]).toarray()
    else:
        review_vect = tfidf.transform([review]).toarray()

    if selectedModel == "Logistic Regression":
        prediction = 'FAKE' if LR_model.predict(review_vect) == 0 else 'REAL'
    elif selectedModel == "Multinomial NB":
        prediction = 'FAKE' if NB_model.predict(review_vect) == 0 else 'REAL'

    return prediction

@app.route('/', methods=['POST'])
def webapp():
    text = request.form['text']
    #prediction = predict(text)

    selectedModel = request.form.get('Models')
    print(str(selectedModel))

    selectedVect = request.form.get('Vectorizers')
    print(str(selectedVect))

    prediction = make_prediction(text, selectedModel, selectedVect)
    
    return render_template('index2.html', text=text, result=prediction)

@app.route('/predict/', methods=['GET','POST'])
def api():
    text = request.args.get("text")
    prediction = make_prediction(text, selectedModel, selectedVect)
    return jsonify(prediction=prediction)

if __name__ == "__main__":
    app.run()