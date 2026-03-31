
import os
from flask import Flask, render_template, request, jsonify
from inference import hybrid_predict

app = Flask(
    __name__,
    template_folder="../frontend",
    static_folder="../frontend",
    static_url_path=""
)

# ---------------- HOME ----------------
@app.route("/")
def home():
    return render_template("index.html")

# ---------------- PREDICT ----------------
@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    result, confidence = hybrid_predict(file)

    return jsonify({
        "prediction": result,
        "confidence": confidence
    })
# ---------------- RUN SERVER ----------------
if __name__ == "__main__":
    app.run(debug=True)