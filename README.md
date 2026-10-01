# Sesi 10 - AI Based Sentiment Review - Flask + Firebase Realtime Database + Firebase Authentication (Login/Register)
# (By Febriyeni Susi - 1076012614908)

Link Railway - Live : https://febriyeni-ai-sentiment-cloud-computing.up.railway.app/

Project web Artificial Intelligence sederhana :
- Flask
- Firebase Realtime Database (BaaS)
- Firebase Authentication (Login/Register)
- Bootstrap 5
- CRUD
- Machine Learning sentiment analysis
- TF-IDF + Logistic Regression
- Untuk menjalankan secara lokal perlu serviceAccountKey.json yang hanya ada pada file zip tidak bisa diupload pada github

Setiap review menyimpan minimal 8 field:
1. name
2. email
3. category
4. review
5. rating
6. sentiment
7. confidence
8. created_at

Tambahan:
- updated_at saat data diubah.

# Install dependency
```bash
pip install -r requirements.txt
```

# Run
```bash
python app.py
```

port and host akan terbuka pada:
http://127.0.0.1:5000


# CRUD
## CREATE
- Nama
- Email
- Kategori
- Rating
- Review

Klik "Analisis & Simpan". Data akan menganalisis AI, mendapat sentiment, mendapatkan confidence, dan disimpan ke Firebase Real-time.

## READ
Semua data Firebase tampil di tabel dashboard.

## UPDATE
Klik Edit lalu ubah data dan klik Update & Analisis Ulang.

## DELETE
Klik Hapus lalu konfirmasi, dan data dihapus dari Firebase.
