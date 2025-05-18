from flask import Flask, request, jsonify, render_template
import joblib
import os

app = Flask(__name__)

model = joblib.load("Spam_Detection.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        msg = data.get("message", "")

        if not msg.strip():
            return jsonify({"error": "Empty message."}), 400
        

        vec = vectorizer.transform([msg])
        if vec.nnz == 0:
            return jsonify({"error": "Message has no recognizable features."}), 400

        pred = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]

        return jsonify({
            "prediction": "Spam" if pred == 1 else "Not Spam",
            "confidence": f"{max(proba) * 100:.2f}%"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
