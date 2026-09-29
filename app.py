import os
from datetime import datetime, timezone

import firebase_admin
from firebase_admin import credentials, db
from flask import Flask, jsonify, redirect, render_template, request, url_for
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

app = Flask(__name__)

SERVICE_ACCOUNT = os.getenv("FIREBASE_SERVICE_ACCOUNT", "serviceAccountKey.json")
DATABASE_URL = os.getenv(
    "FIREBASE_DATABASE_URL",
    "https://testing-d431b-default-rtdb.firebaseio.com/"
)

if not firebase_admin._apps:
    if not os.path.exists(SERVICE_ACCOUNT):
        raise FileNotFoundError("Firebase service account file not found.")
    cred = credentials.Certificate(SERVICE_ACCOUNT)
    firebase_admin.initialize_app(cred, {"databaseURL": DATABASE_URL})

reviews_ref = db.reference("ai_reviews")

TRAIN_DATA = [
    ("aplikasinya sangat bagus dan mudah digunakan", "Positif"),
    ("saya sangat puas dengan pelayanan ini", "Positif"),
    ("fiturnya keren dan membantu pekerjaan saya", "Positif"),
    ("produk ini luar biasa saya suka sekali", "Positif"),
    ("pelayanannya cepat dan ramah", "Positif"),
    ("hasilnya sangat memuaskan", "Positif"),
    ("website ini nyaman digunakan", "Positif"),
    ("saya senang menggunakan aplikasi ini", "Positif"),
    ("sangat baik dan saya rekomendasikan", "Positif"),
    ("pengalaman yang menyenangkan", "Positif"),
    ("aplikasi ini membantu sekali, terima kasih", "Positif"),
    ("kualitasnya bagus dan harga terjangkau", "Positif"),
    ("saya akan menggunakan lagi layanan ini", "Positif"),
    ("respon cepat dan solutif", "Positif"),
    ("desainnya menarik dan modern", "Positif"),
    ("sangat memuaskan, tidak ada kendala", "Positif"),
    ("top banget, lanjutkan!", "Positif"),
    ("mantap, saya puas sekali", "Positif"),
    ("pelayanan ramah dan profesional", "Positif"),
    ("fitur lengkap dan mudah dipahami", "Positif"),
    ("sangat membantu pekerjaan sehari-hari", "Positif"),
    ("aplikasi terbaik yang pernah saya pakai", "Positif"),
    ("prosesnya cepat dan tidak ribet", "Positif"),
    ("saya suka tampilannya yang bersih", "Positif"),
    ("harga sesuai kualitas, puas", "Positif"),
    ("customer service-nya sangat membantu", "Positif"),
    ("pengiriman cepat dan aman", "Positif"),
    ("produk original, saya senang", "Positif"),
    ("pengalaman belanja yang menyenangkan", "Positif"),
    ("terima kasih, pelayanannya memuaskan", "Positif"),

    ("aplikasinya buruk dan sering error", "Negatif"),
    ("saya sangat kecewa dengan pelayanan ini", "Negatif"),
    ("fiturnya tidak berguna dan sulit digunakan", "Negatif"),
    ("produk ini jelek sekali", "Negatif"),
    ("pelayanannya lambat dan mengecewakan", "Negatif"),
    ("hasilnya sangat buruk", "Negatif"),
    ("website ini tidak nyaman", "Negatif"),
    ("saya tidak suka menggunakan aplikasi ini", "Negatif"),
    ("pengalaman yang mengecewakan", "Negatif"),
    ("banyak masalah dan error", "Negatif"),
    ("aplikasi ini sering crash dan lag", "Negatif"),
    ("pelayanan sangat lambat, tidak profesional", "Negatif"),
    ("produk rusak saat diterima", "Negatif"),
    ("saya menyesal membeli produk ini", "Negatif"),
    ("tidak sesuai deskripsi, sangat mengecewakan", "Negatif"),
    ("fitur tidak berfungsi dengan baik", "Negatif"),
    ("sulit digunakan dan membingungkan", "Negatif"),
    ("customer service tidak responsif", "Negatif"),
    ("pengiriman sangat lama", "Negatif"),
    ("kualitas buruk, tidak worth it", "Negatif"),
    ("aplikasi ini payah, banyak bug", "Negatif"),
    ("saya benci menggunakan aplikasi ini", "Negatif"),
    ("pelayanan buruk dan tidak ramah", "Negatif"),
    ("website sering down dan tidak bisa diakses", "Negatif"),
    ("produk tidak sesuai harapan", "Negatif"),
    ("harga mahal tapi kualitas jelek", "Negatif"),
    ("sangat lambat dan tidak efisien", "Negatif"),
    ("error terus, tidak bisa dipakai", "Negatif"),
    ("saya komplain tapi tidak ditanggapi", "Negatif"),
    ("pengalaman buruk, tidak akan kembali", "Negatif"),

    ("aplikasinya biasa saja", "Netral"),
    ("pelayanannya cukup", "Netral"),
    ("fiturnya lumayan", "Netral"),
    ("produk ini standar", "Netral"),
    ("pengalamannya tidak terlalu buruk", "Netral"),
    ("hasilnya biasa saja", "Netral"),
    ("website ini cukup mudah", "Netral"),
    ("saya tidak punya pendapat khusus", "Netral"),
    ("biasa aja, tidak ada yang istimewa", "Netral"),
    ("cukup lah, sesuai harga", "Netral"),
    ("standar, tidak lebih tidak kurang", "Netral"),
    ("lumayan, tapi bisa lebih baik", "Netral"),
    ("netral saja, tidak ada masalah", "Netral"),
    ("oke lah, tidak mengecewakan", "Netral"),
    ("sama seperti aplikasi lain", "Netral"),
    ("tidak terlalu suka, tidak terlalu benci", "Netral"),
    ("biasa, tidak ada yang perlu dibahas", "Netral"),
    ("cukup memadai untuk kebutuhan saya", "Netral"),
    ("standar industri, tidak ada kejutan", "Netral"),
    ("sedang-sedang saja", "Netral"),
    ("tidak buruk, tidak juga bagus", "Netral"),
    ("lumayan untuk pemula", "Netral"),
    ("bisa dibilang cukup", "Netral"),
    ("tidak ada komentar khusus", "Netral"),
    ("netral, tidak ada keluhan berarti", "Netral"),
    ("sesuai ekspektasi, tidak lebih", "Netral"),
    ("biasa aja sih menurut saya", "Netral"),
    ("cukup oke, tapi tidak istimewa", "Netral"),
    ("standar banget, tidak ada yang wow", "Netral"),
    ("ya, lumayan lah", "Netral"),
]

TRAIN_TEXTS = [text for text, _ in TRAIN_DATA]
TRAIN_LABELS = [label for _, label in TRAIN_DATA]

ai_model = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 3),
        sublinear_tf=True,
        min_df=1,
        max_df=0.95,
        lowercase=True,
    )),
    ("classifier", LogisticRegression(
        max_iter=2000,
        class_weight="balanced", 
        C=2.0,
        solver="lbfgs",
    ))
])

ai_model.fit(TRAIN_TEXTS, TRAIN_LABELS)


def predict_sentiment(text):
    """Prediksi sentimen dengan confidence + fallback rule-based untuk Netral."""
    if not text or not text.strip():
        return "Netral", 0.0

    text_lower = text.lower().strip()

    netral_markers = [
        "biasa saja", "biasa aja", "cukup", "lumayan", "standar",
        "tidak ada masalah", "tidak terlalu", "sedang", "oke lah",
        "tidak buruk", "tidak istimewa", "netral", "b aja"
    ]
    negatif_kuat = ["buruk", "jelek", "kecewa", "error", "parah", "benci", "rusak"]
    positif_kuat = ["bagus", "suka", "puas", "mantap", "luar biasa", "keren", "senang"]

    has_neg_kuat = any(w in text_lower for w in negatif_kuat)
    has_pos_kuat = any(w in text_lower for w in positif_kuat)

    if any(m in text_lower for m in netral_markers) and not has_neg_kuat and not has_pos_kuat:
        probs = ai_model.predict_proba([text])[0]
        classes = list(ai_model.classes_)
        idx_netral = classes.index("Netral") if "Netral" in classes else 0
        return "Netral", round(float(probs[idx_netral] * 100), 2)

    probabilities = ai_model.predict_proba([text])[0]
    classes = list(ai_model.classes_)
    best_index = probabilities.argmax()
    sentiment = classes[best_index]
    confidence = float(probabilities[best_index] * 100)

    return sentiment, round(confidence, 2)


def get_all_reviews():
    data = reviews_ref.get() or {}
    rows = []
    for review_id, item in data.items():
        if isinstance(item, dict):
            item = dict(item)
            item["id"] = review_id
            rows.append(item)
    rows.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return rows


@app.route("/")
def index():
    reviews = get_all_reviews()
    counts = {
        "total": len(reviews),
        "positif": sum(1 for x in reviews if x.get("sentiment") == "Positif"),
        "negatif": sum(1 for x in reviews if x.get("sentiment") == "Negatif"),
        "netral": sum(1 for x in reviews if x.get("sentiment") == "Netral"),
    }
    return render_template("index.html", reviews=reviews, counts=counts)


@app.route("/create", methods=["POST"])
def create():
    name = request.form.get("name", "").strip()
    email = request.form.get("email", "").strip()
    category = request.form.get("category", "").strip()
    review = request.form.get("review", "").strip()
    rating = request.form.get("rating", "").strip()

    if not name or not email or not category or not review or not rating:
        return redirect(url_for("index"))

    sentiment, confidence = predict_sentiment(review)

    record = {
        "name": name,
        "email": email,
        "category": category,
        "review": review,
        "rating": int(rating),
        "sentiment": sentiment,
        "confidence": confidence,
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    reviews_ref.push(record)
    return redirect(url_for("index"))


@app.route("/edit/<review_id>")
def edit(review_id):
    item = reviews_ref.child(review_id).get()
    if not item:
        return redirect(url_for("index"))
    item["id"] = review_id
    return render_template("edit.html", review=item)


@app.route("/update/<review_id>", methods=["POST"])
def update(review_id):
    existing = reviews_ref.child(review_id).get()
    if not existing:
        return redirect(url_for("index"))

    name = request.form.get("name", "").strip()
    email = request.form.get("email", "").strip()
    category = request.form.get("category", "").strip()
    review = request.form.get("review", "").strip()
    rating = request.form.get("rating", "").strip()

    if not name or not email or not category or not review or not rating:
        return redirect(url_for("edit", review_id=review_id))

    sentiment, confidence = predict_sentiment(review)

    updated_record = {
        "name": name,
        "email": email,
        "category": category,
        "review": review,
        "rating": int(rating),
        "sentiment": sentiment,
        "confidence": confidence,
        "created_at": existing.get("created_at", datetime.now(timezone.utc).isoformat()),
        "updated_at": datetime.now(timezone.utc).isoformat()
    }
    reviews_ref.child(review_id).update(updated_record)
    return redirect(url_for("index"))


@app.route("/delete/<review_id>", methods=["POST"])
def delete(review_id):
    reviews_ref.child(review_id).delete()
    return redirect(url_for("index"))


@app.route("/api/reviews", methods=["GET"])
def api_get_reviews():
    return jsonify(get_all_reviews())


@app.route("/api/reviews", methods=["POST"])
def api_create_review():
    data = request.get_json(silent=True) or {}
    required = ["name", "email", "category", "review", "rating"]
    if any(not data.get(field) for field in required):
        return jsonify({"error": "All fields are required"}), 400

    sentiment, confidence = predict_sentiment(data["review"])
    record = {
        "name": data["name"],
        "email": data["email"],
        "category": data["category"],
        "review": data["review"],
        "rating": int(data["rating"]),
        "sentiment": sentiment,
        "confidence": confidence,
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    new_ref = reviews_ref.push(record)
    return jsonify({"id": new_ref.key, **record}), 201


@app.route("/api/reviews/<review_id>", methods=["PUT"])
def api_update_review(review_id):
    existing = reviews_ref.child(review_id).get()
    if not existing:
        return jsonify({"error": "Review not found"}), 404

    data = request.get_json(silent=True) or {}
    review_text = data.get("review", existing.get("review", ""))
    sentiment, confidence = predict_sentiment(review_text)

    updated = {
        "name": data.get("name", existing.get("name", "")),
        "email": data.get("email", existing.get("email", "")),
        "category": data.get("category", existing.get("category", "")),
        "review": review_text,
        "rating": int(data.get("rating", existing.get("rating", 0))),
        "sentiment": sentiment,
        "confidence": confidence,
        "created_at": existing.get("created_at"),
        "updated_at": datetime.now(timezone.utc).isoformat()
    }
    reviews_ref.child(review_id).update(updated)
    return jsonify({"id": review_id, **updated})


@app.route("/api/reviews/<review_id>", methods=["DELETE"])
def api_delete_review(review_id):
    reviews_ref.child(review_id).delete()
    return jsonify({"message": "Deleted successfully"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)