import os
import json
from datetime import datetime, timezone
from functools import wraps

import requests
from flask import (
    Flask, jsonify, redirect, render_template, request, session, url_for
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = Flask(__name__)

app.secret_key = os.getenv(
    "FLASK_SECRET_KEY"
)

FIREBASE_DB_URL = os.getenv("FIREBASE_DATABASE_URL")
FIREBASE_DB_SECRET = os.getenv("FIREBASE_DATABASE_SECRET")
FIREBASE_API_KEY = os.getenv("FIREBASE_WEB_API_KEY")
FIREBASE_AUTH_BASE_URL = "https://identitytoolkit.googleapis.com/v1/accounts:"

APP_BASE_URL = os.getenv("APP_BASE_URL", "").rstrip("/")

if FIREBASE_DB_URL and FIREBASE_DB_URL.endswith("/"):
    FIREBASE_DB_URL = FIREBASE_DB_URL[:-1]

if not FIREBASE_DB_URL:
    raise ValueError(
        "FIREBASE_DATABASE_URL tidak diset. "
    )
if not FIREBASE_DB_SECRET:
    raise ValueError(
        "FIREBASE_DATABASE_SECRET tidak diset. "
    )

print("Konfigurasi Firebase REST API siap")
print(f"   Database URL: {FIREBASE_DB_URL}")


def firebase_get(path):
    url = f"{FIREBASE_DB_URL}/{path}.json?auth={FIREBASE_DB_SECRET}"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error firebase_get({path}): {e}")
        return None


def firebase_set(path, data):
    url = f"{FIREBASE_DB_URL}/{path}.json?auth={FIREBASE_DB_SECRET}"
    try:
        response = requests.put(url, json=data, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error firebase_set({path}): {e}")
        return None


def firebase_push(path, data):
    url = f"{FIREBASE_DB_URL}/{path}.json?auth={FIREBASE_DB_SECRET}"
    try:
        response = requests.post(url, json=data, timeout=15)
        response.raise_for_status()
        result = response.json()
        return result.get("name") if isinstance(result, dict) else None
    except requests.exceptions.RequestException as e:
        print(f"Error firebase_push({path}): {e}")
        return None


def firebase_update(path, data):
    url = f"{FIREBASE_DB_URL}/{path}.json?auth={FIREBASE_DB_SECRET}"
    try:
        response = requests.patch(url, json=data, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error firebase_update({path}): {e}")
        return None


def firebase_delete(path):
    url = f"{FIREBASE_DB_URL}/{path}.json?auth={FIREBASE_DB_SECRET}"
    try:
        response = requests.delete(url, timeout=15)
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error firebase_delete({path}): {e}")
        return False


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
print("Model ML berhasil dilatih")


def predict_sentiment(text):
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
    data = firebase_get("ai_reviews") or {}
    rows = []
    for review_id, item in data.items():
        if isinstance(item, dict):
            item = dict(item)
            item["id"] = review_id
            rows.append(item)
    rows.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return rows


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "id_token" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function


def firebase_sign_up(email, password):
    url = f"{FIREBASE_AUTH_BASE_URL}signUp?key={FIREBASE_API_KEY}"
    payload = {"email": email, "password": password, "returnSecureToken": True}
    return requests.post(url, json=payload, timeout=15)


def firebase_sign_in(email, password):
    url = f"{FIREBASE_AUTH_BASE_URL}signInWithPassword?key={FIREBASE_API_KEY}"
    payload = {"email": email, "password": password, "returnSecureToken": True}
    return requests.post(url, json=payload, timeout=15)


def firebase_update_profile(id_token, display_name):
    url = f"{FIREBASE_AUTH_BASE_URL}update?key={FIREBASE_API_KEY}"
    payload = {"idToken": id_token, "displayName": display_name, "returnSecureToken": True}
    return requests.post(url, json=payload, timeout=15)


def firebase_send_verification_email(id_token, continue_url=None):
    url = f"{FIREBASE_AUTH_BASE_URL}sendOobCode?key={FIREBASE_API_KEY}"
    payload = {
        "requestType": "VERIFY_EMAIL",
        "idToken": id_token
    }
    if continue_url:
        payload["continueUrl"] = continue_url
    return requests.post(url, json=payload, timeout=15)


def firebase_lookup(id_token):
    url = f"{FIREBASE_AUTH_BASE_URL}lookup?key={FIREBASE_API_KEY}"
    payload = {"idToken": id_token}
    return requests.post(url, json=payload, timeout=15)


def translate_auth_error(msg):
    mapping = {
        "EMAIL_NOT_FOUND": "Email tidak terdaftar. Silakan register terlebih dahulu.",
        "INVALID_PASSWORD": "Password salah. Silakan periksa kembali.",
        "INVALID_LOGIN_CREDENTIALS": "Email atau password salah.",
        "INVALID_EMAIL": "Format email tidak valid.",
        "EMAIL_EXISTS": "Email sudah terdaftar. Silakan login.",
        "USER_DISABLED": "Akun ini dinonaktifkan.",
        "TOO_MANY_ATTEMPTS_TRY_LATER": "Terlalu banyak percobaan. Coba lagi nanti.",
        "WEAK_PASSWORD": "Password terlalu lemah. Gunakan minimal 6 karakter.",
        "MISSING_EMAIL": "Email wajib diisi.",
        "MISSING_PASSWORD": "Password wajib diisi.",
        "OPERATION_NOT_ALLOWED": "Metode login ini belum diaktifkan.",
        "INVALID_ID_TOKEN": "Sesi tidak valid. Silakan login ulang.",
        "USER_NOT_FOUND": "Akun tidak ditemukan.",
        "EXPIRED_OOB_CODE": "Link verifikasi sudah kedaluwarsa. Minta link baru.",
        "INVALID_OOB_CODE": "Link verifikasi tidak valid atau sudah digunakan.",
    }
    return mapping.get(msg, f"Terjadi kesalahan: {msg}")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")

        if not name or not email or not password:
            return render_template("register.html", error="Semua field wajib diisi.")
        if password != confirm:
            return render_template("register.html", error="Password dan konfirmasi tidak cocok.")
        if len(password) < 6:
            return render_template("register.html", error="Password minimal 6 karakter.")

        try:
            response = firebase_sign_up(email, password)
            if response.status_code != 200:
                msg = response.json().get("error", {}).get("message", "REGISTRATION_FAILED")
                return render_template("register.html", error=translate_auth_error(msg))

            data = response.json()
            id_token = data["idToken"]
            local_id = data["localId"]

            firebase_set(f"users/{local_id}", {
                "name": name,
                "email": email,
                "email_verified": False,
                "created_at": datetime.now(timezone.utc).isoformat()
            })

            try:
                firebase_update_profile(id_token, name)
            except Exception:
                pass

            continue_url = f"{APP_BASE_URL}/login?verified=1" if APP_BASE_URL else None

            try:
                verify_resp = firebase_send_verification_email(id_token, continue_url)
                if verify_resp.status_code != 200:
                    err = verify_resp.json().get("error", {}).get("message", "SEND_EMAIL_FAILED")
                    print(f"Gagal kirim email verifikasi: {err}")
                    # Tetap lanjut, tapi kasih tahu user
                    return render_template(
                        "register.html",
                        error=f"Akun dibuat, tetapi gagal mengirim email verifikasi: {translate_auth_error(err)}. "
                              f"Silakan coba login lalu klik 'Kirim ulang verifikasi'."
                    )
            except requests.exceptions.RequestException as e:
                print(f"Error koneksi saat kirim email: {e}")
                return render_template(
                    "register.html",
                    error="Akun dibuat, tetapi gagal mengirim email verifikasi. "
                          "Silakan coba login lalu klik 'Kirim ulang verifikasi'."
                )


            return render_template(
                "register.html",
                verification_pending=True,
                success=f"Akun berhasil dibuat! Link verifikasi telah dikirim ke {email}.",
                email_for_resend=email
            )

        except requests.exceptions.RequestException:
            return render_template("register.html", error="Gagal terhubung ke server autentikasi.")

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    just_verified = request.args.get("verified") == "1"
    success_msg = None
    if just_verified:
        success_msg = "Email Anda telah diverifikasi! Silakan login sekarang."

    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        if not email or not password:
            return render_template(
                "login.html",
                error="Email dan password wajib diisi.",
                success=success_msg
            )

        try:
            response = firebase_sign_in(email, password)

            if response.status_code != 200:
                msg = response.json().get("error", {}).get("message", "LOGIN_FAILED")
                return render_template(
                    "login.html",
                    error=translate_auth_error(msg),
                    success=success_msg
                )

            data = response.json()
            local_id = data["localId"]
            id_token = data["idToken"]

            email_verified = False
            try:
                lookup_resp = firebase_lookup(id_token)
                if lookup_resp.status_code == 200:
                    users = lookup_resp.json().get("users", [])
                    if users:
                        email_verified = users[0].get("emailVerified", False)
                        print(f"[LOGIN] User {email} → emailVerified={email_verified}")
                else:
                    email_verified = data.get("emailVerified", False)
                    print(f"[LOGIN] Lookup failed, fallback ke signIn value: {email_verified}")
            except Exception as e:
                print(f"[LOGIN] Error saat lookup: {e}")
                email_verified = data.get("emailVerified", False)

            if not email_verified:
                return render_template(
                    "login.html",
                    error="Email Anda belum diverifikasi. Silakan cek inbox atau folder spam "
                          "lalu klik link verifikasi. Jika tidak menerima email, "
                          "klik tombol 'Kirim Ulang Verifikasi' di bawah.",
                    show_resend=True,
                    email_for_resend=email
                )

            profile = firebase_get(f"users/{local_id}") or {}
            name = profile.get("name") or data.get("displayName") or email.split("@")[0]

            if not profile.get("email_verified"):
                firebase_update(f"users/{local_id}", {
                    "email_verified": True,
                    "verified_at": datetime.now(timezone.utc).isoformat()
                })

            session["id_token"] = id_token
            session["local_id"] = local_id
            session["email"] = data["email"]
            session["name"] = name

            return redirect(url_for("index"))

        except requests.exceptions.RequestException:
            return render_template(
                "login.html",
                error="Gagal terhubung ke server autentikasi.",
                success=success_msg
            )

    return render_template("login.html", success=success_msg)

@app.route("/resend-verification", methods=["POST"])
def resend_verification():
    email = request.form.get("email", "").strip()
    password = request.form.get("password", "").strip()
    from_register = request.form.get("from_register") == "1"

    target_template = "register.html" if from_register else "login.html"
    extra_ctx = {"verification_pending": True} if from_register else {"show_resend": True}

    if not email or not password:
        return render_template(
            target_template,
            error="Untuk mengirim ulang verifikasi, isi email dan password Anda.",
            email_for_resend=email,
            **extra_ctx
        )

    try:
        signin_resp = firebase_sign_in(email, password)
        if signin_resp.status_code != 200:
            msg = signin_resp.json().get("error", {}).get("message", "LOGIN_FAILED")
            return render_template(
                target_template,
                error=translate_auth_error(msg),
                email_for_resend=email,
                **extra_ctx
            )

        data = signin_resp.json()
        id_token = data["idToken"]

        email_verified = False
        try:
            lookup_resp = firebase_lookup(id_token)
            if lookup_resp.status_code == 200:
                users = lookup_resp.json().get("users", [])
                if users:
                    email_verified = users[0].get("emailVerified", False)
        except Exception:
            email_verified = data.get("emailVerified", False)

        if email_verified:
            return render_template(
                "login.html",
                success="✅ Email Anda sudah diverifikasi! Silakan login sekarang.",
                email_for_resend=email
            )

        continue_url = f"{APP_BASE_URL}/login?verified=1" if APP_BASE_URL else None
        resp = firebase_send_verification_email(id_token, continue_url)

        if resp.status_code == 200:
            return render_template(
                target_template,
                success=f"Email verifikasi baru telah dikirim ke {email}.",
                email_for_resend=email,
                **extra_ctx
            )
        else:
            err = resp.json().get("error", {}).get("message", "SEND_EMAIL_FAILED")
            return render_template(
                target_template,
                error=translate_auth_error(err),
                email_for_resend=email,
                **extra_ctx
            )

    except requests.exceptions.RequestException:
        return render_template(
            target_template,
            error="Gagal terhubung ke server autentikasi.",
            email_for_resend=email,
            **extra_ctx
        )


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/profile")
@login_required
def profile():
    return render_template(
        "profile.html",
        email=session.get("email"),
        name=session.get("name"),
        local_id=session.get("local_id"),
    )


@app.route("/")
@login_required
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
@login_required
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
        "user_id": session.get("local_id"),
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    firebase_push("ai_reviews", record)
    return redirect(url_for("index"))


@app.route("/edit/<review_id>")
@login_required
def edit(review_id):
    item = firebase_get(f"ai_reviews/{review_id}")
    if not item:
        return redirect(url_for("index"))
    item["id"] = review_id
    return render_template("edit.html", review=item)


@app.route("/update/<review_id>", methods=["POST"])
@login_required
def update(review_id):
    existing = firebase_get(f"ai_reviews/{review_id}")
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
    firebase_update(f"ai_reviews/{review_id}", updated_record)
    return redirect(url_for("index"))


@app.route("/delete/<review_id>", methods=["POST"])
@login_required
def delete(review_id):
    firebase_delete(f"ai_reviews/{review_id}")
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
    new_key = firebase_push("ai_reviews", record)
    return jsonify({"id": new_key, **record}), 201


@app.route("/api/reviews/<review_id>", methods=["PUT"])
def api_update_review(review_id):
    existing = firebase_get(f"ai_reviews/{review_id}")
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
    firebase_update(f"ai_reviews/{review_id}", updated)
    return jsonify({"id": review_id, **updated})


@app.route("/api/reviews/<review_id>", methods=["DELETE"])
def api_delete_review(review_id):
    firebase_delete(f"ai_reviews/{review_id}")
    return jsonify({"message": "Deleted successfully"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False)
