from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load the pre-trained model and vectorizer
dt_classifier = joblib.load('customerReview/sentiment_model.pkl')
tfidf = joblib.load('customerReview/tfidf_vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.json['review']
    
    # Preprocess and vectorize the input review
    review_tfidf = tfidf.transform([review])
    
    # Predict sentiment using the trained model
    prediction = dt_classifier.predict(review_tfidf)
    
    # Return the result
    sentiment = 'positive' if prediction[0] == 1 else 'negative'
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)
