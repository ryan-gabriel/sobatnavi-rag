# 📄 DOKUMEN SYSTEM REQUIREMENTS & TEST CASES: SOBATNAVI AI AGENT

## 🎯 A. CORE REQUIREMENTS (Persyaratan Sistem)

**1. Aturan Format & Output AI (Strict Constraints)**

* Sistem harus selalu mengembalikan respons dalam format **JSON murni** yang divalidasi oleh Pydantic (`FinalAIResponse`).
* LLM dilarang keras menggunakan *markdown formatting* (seperti ````json`).
* LLM **DILARANG HALUSINASI** (mengarang) nama tempat, `poi_id`, atau koordinat. Semua data tempat wisata, hotel, dan restoran harus berasal dari pemanggilan *Tools* (Database).

**2. Pendeteksian Niat Pengguna (Intent Detection)**

* Jika *user* hanya menyapa atau ngobrol ("Hai", "Terima kasih"): Harus merespons dengan `response_type: "chat"` tanpa memanggil *tool* database.
* Jika *user* hanya minta rekomendasi ("Apa pantai yang bagus?"): Harus merespons dengan `response_type: "recommendation"` dan memanggil *tool* pencarian, tanpa membuat jadwal harian.
* Jika *user* minta dibuatkan *itinerary*:
* **Mode General:** Langsung buatkan jadwal.
* **Mode Deep Research:** Tahan pembuatan jadwal. Pastikan 5 variabel terpenuhi dari riwayat chat (Lokasi, Tanggal, Budget, Companion, Pace). Jika belum lengkap, balas dengan `response_type: "clarifying"` dan tanya variabel yang kurang.



**3. Algoritma Mesin Rekomendasi (TOPSIS Multi-Dimensi & DBSCAN)**

* **Semantic Search:** Pencarian tempat tidak menggunakan *keyword match* biasa, melainkan Vector RAG (Semantic Search) ke Supabase.
* **Clustering (DBSCAN):** Tempat-tempat dalam satu hari harus berdekatan secara jarak (menggunakan DBSCAN radius ~15km untuk filter spasial bumi).
* **Dynamic Multi-Dimensional TOPSIS:** Perangkingan tempat tidak boleh hanya mengandalkan kolom statis SQL (`rating` & `user_rating_count`). Sistem **WAJIB** mengekstrak parameter dari dalam kolom JSONB `metadata`, khususnya pada *path* `metadata.topsis_features`.
* **POI:** Bobot difokuskan pada `rating`, `popularity`, `price_value`, `strategic_score`, dan `visual_interest_index`.
* **Hotel:** Bobot difokuskan pada `rating`, `comfort_index`, `amenity_density`, dan `accessibility_index`.
* **Restoran:** Bobot difokuskan pada `rating`, `menu_variety_index`, `ambience_score`, dan `payment_modern_index`.


* **TOPSIS Fallback Logic:** Jika suatu data di database belum memiliki *key* `topsis_features` di metadatanya, sistem Python harus memiliki logika *fallback* untuk menghitung nilai sementara dari *raw metadata* (misalnya menghitung panjang array `amenities` atau `serves`) agar sistem tidak *crash*.

**4. Manajemen Rute & Hotel (Tools Engineering)**

* **Base Hotel (Anchor):** Dalam satu *itinerary*, AI hanya boleh memilih **SATU** hotel untuk seluruh perjalanan (tidak boleh pindah hotel setiap hari).
* **Batch Routing (TomTom):** Jarak dan waktu tempuh antar tempat dalam satu hari harus dihitung menggunakan *tool* `calculate_batch_routes` secara kolektif (satu panggilan per hari, bukan per tempat), dengan memasukkan parameter zona penghindaran Odalan.

**5. Fitur CRUD & Sinkronisasi Database**

* **Auto-Save:** Setiap kali *itinerary* baru berhasil dibuat, *backend* Python otomatis menyimpannya ke tabel `user_itineraries` menggunakan `user_id` dari *Bearer Token*.
* **Edit via Chat (AI):** Jika ada `current_itinerary` dan `itinerary_id` pada *request*, AI bertugas memodifikasi JSON lama menggunakan tool `search_specific_place`. Setelah selesai, *backend* otomatis melakukan *UPDATE* data tersebut di database (Auto-Save Revisi).
* **Edit Manual (UI):** Terdapat *endpoint* REST API terpisah (`GET`, `PUT`, `DELETE`) agar *user* bisa mengedit *itinerary* langsung dari *User Interface* tanpa intervensi AI.

---

## 🧪 B. DAFTAR TEST CASES (Skenario Pengujian)

### Kategori 1: Obrolan Biasa & Rekomendasi (Intent Handling)

* **TC-01 (Greeting):** Jika *request user* adalah "Hai Heidi", respons bertipe `"chat"` berisi sapaan ramah, dan AI **TIDAK BOLEH** memanggil *tools* pencarian.
* **TC-02 (Rekomendasi Murni):** Jika *request user* adalah "Kasih rekomendasi pantai di Bali dong", respons bertipe `"recommendation"`. AI menampilkan daftarnya tetapi **TIDAK BOLEH** menyusunnya ke dalam *array* `itinerary_days`.

### Kategori 2: Mode Pembuatan Itinerary & Penanganan Data Kosong

* **TC-03 (Deep Research - Kurang Data):** Jika *request* "Buatkan jadwal di Ubud" (mode `deep_research`), respons bertipe `"clarifying"`. AI bertanya parameter yang kurang (budget, pace, dll) dan menahan eksekusi *tools*.
* **TC-04 (General Mode - Empty Data):** Jika *user* mencari "Itinerary ke Planet Mars", AI mencoba mencari minimal 2 kali dengan kata kunci berbeda. Jika tetap kosong, respons bertipe `"clarifying"` (memberitahu data tidak ada), dan **TIDAK BOLEH** menggunakan tipe respons buatan seperti `"ERROR"`.

### Kategori 3: Kualitas Data Mesin Rekomendasi (TOPSIS & Rute)

* **TC-05 (TOPSIS Dynamic Extraction):** Saat `cluster_and_rank_pois` dijalankan untuk data Restoran, matriks NumPy harus terisi dengan minimal 6 fitur (seperti `menu_variety_index`, `ambience_score`, dll) yang diekstrak dari `metadata.topsis_features`, bukan hanya 2 fitur. Restoran dengan skor *ambience* tinggi harus menang *ranking* dibanding warung biasa walau rating-nya sama.
* **TC-06 (TOPSIS Fallback Resilience):** Jika ada satu baris data hotel yang JSON `metadata`-nya benar-benar kosong (`{}`), fungsi TOPSIS **TIDAK BOLEH** melempar *KeyError* atau *crash*. Sistem harus memberikan nilai *default* (misal 0 atau nilai tengah) pada baris data tersebut dan kalkulasi tetap berjalan.
* **TC-07 (Hotel Anchor):** Di dalam JSON *response* itinerary 3 hari, variabel `base_hotel` hanya terisi satu nama hotel, dan hotel **TIDAK BOLEH** muncul lagi di dalam daftar kunjungan harian (`itinerary_days`).
* **TC-08 (Batch Routing):** Variabel `distance_to_next_km` dan `travel_time_to_next_mins` pada setiap tempat wisata harus terisi angka yang didapat dari *tool* `calculate_batch_routes`.

### Kategori 4: Fitur Editing & Sinkronisasi Database

* **TC-09 (Hapus Tempat via AI):** Jika *user* pesan "Hapus tempat pertama di hari ke-2" (disertai `current_itinerary` & `itinerary_id`), respons bertipe `"itinerary"`. JSON hasilnya membuang tempat tersebut, lalu *backend* Python otomatis men-*trigger* `UPDATE` ke tabel `user_itineraries`.
* **TC-10 (Ganti Tempat via AI):** Jika *user* pesan "Ganti makan siang di hari 1 dengan Bebek Tepi Sawah", AI harus memanggil `search_specific_place`, menyisipkan ID/Koordinat aslinya, lalu *backend* Python otomatis menyimpannya ke database.

### Kategori 5: Auth & Security

* **TC-11 (Unauthorized Access):** Jika *request* ke `/api/chat` tidak memiliki *Bearer Token*, API menolak dengan HTTP Status `401 Unauthorized`.
* **TC-12 (Data Ownership):** Jika *itinerary* berhasil dibuat/diedit, data tersebut harus tersimpan di `user_itineraries` dengan kolom `user_id` yang cocok dengan token pengguna, dan `is_public` tersetting `False`.