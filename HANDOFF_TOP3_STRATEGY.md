# HANDOFF — Solar Filament Segmentation 2026: strategi menuju Top 3

Dokumen serah-terima untuk sesi Claude Code khusus. Baca seluruhnya sebelum menjalankan apa pun.
Disusun 2026-10-14 dari `report/report.md`, `README.md`, dan git log repo ini (45 commit, terakhir 2026-10-03).
Label: **[fakta]** = tercatat di repo/report; **[inferensi]** = kesimpulan penyusun, belum diuji; **[tidak diketahui]** = harus dicek.

---

## 0. Ringkasan satu paragraf

Pipeline dua tahap (5-fold U-Net ResNet34 + crop refiner + near-limb filter + LightGBM instance scorer) mendapat **hold-out PQ 0.4427**, **OOF PQ 0.409**, tetapi **LB publik hanya 0.37**. Papan "valid" berakhir di **0.41** (2 tim), 0.40 (11 tim), 0.39 (21 tim). Di atasnya ada 52 tim di 0.49–0.62 yang dianggap invalid oleh organizer. Peringkat kita ~96 dari ~610 tim valid (report.md:111 mencatat rank 167 dari 806 semua tim pada 0.37). Untuk Top 3 di LB valid diperlukan **≥0.41 (+0.04)**. Hampir semua lever pasca-proses sudah dicoba dan bernilai nol; bukti menunjuk ke **kualitas stage-A** dan ke **gap 0.04–0.05 antara OOF/hold-out dan LB yang belum dijelaskan**. Evaluasi final organizer = **kuantitatif + kualitatif**, jadi strategi dibagi dua track: (A) naikkan skor, (B) menangkan penilaian kualitatif.

## 1. Kompetisi dan metrik

- IEEE BigData Cup 2026 / Kaggle "Solar Filament Segmentation Challenge 2026"; data MAGFiLO (H-alpha GONG, 2048×2048). [fakta]
- Train 707 gambar beranotasi, test 180 gambar. 296 gambar punya ≥2 pass anotator (~42%); ~58% hanya 1 pass. [fakta]
- Metrik: PQ = Σ IoU(TP) / (|TP| + 0.5|FP| + 0.5|FN|), TP jika IoU>0.5, dihitung per pass anotator lalu di-pool. Implementasi: `filament_metric.py`. [fakta]
- Konsekuensi: menyimpan satu instance hanya menguntungkan jika expected matched IoU > PQ/2 (~0.197). [fakta, dasar scorer]
- Hold-out: 50 gambar, seed 1234, tidak dipakai training/seleksi checkpoint/refiner. SE bootstrap ≈ 0.017 (resolusi rendah). OOF: 657 gambar, tiap gambar dinilai fold model yang tidak melihatnya. [fakta]

## 2. Status pipeline (rilis saat ini)

1. **Stage-A**: 5 U-Net (smp, ResNet34 ImageNet), 1 per fold atas 657 gambar pool, 30 epoch × 220 step × batch 8, patch 512; TTA dihedral/flip. Kandidat = komponen terhubung `prob>0.4`, area 400–40000 px, mean prob ≥0.7.
2. **Crop refiner**: crop 1.5× bbox → 512², input `[crop, CLAHE, seed]` seed = bbox + padding 40%; menggambar ulang mask. Satu-satunya perubahan dengan efek terukur: **+0.0177** hold-out PQ.
3. **Near-limb filter**: buang instance dengan centroid > 0.88 radius disk (Otsu). OOF +0.0065, hold-out +0.0059. Dapat dimatikan dengan `Pipeline(limb_drop=None)`.
4. **Instance scorer** (LightGBM, 36 fitur, dilatih pada 4,694 instance OOF, threshold tetap PQ/2=0.197): OOF 0.3944→0.4088 (+0.0144, CI95 +0.011..+0.018); hold-out 0.4383→0.4427 (n.s.). LB: 0.36→0.37.

Tabel skor (hold-out / LB): proxy awal 0.3754 / 0.33; mix R34+R50 0.4030 / 0.34; refiner 0.4325 / 0.36; +scorer 0.4427 / 0.37. Urutan hold-out konsisten dengan urutan LB. [fakta]

## 3. Peta lever: apa yang sudah dicoba

| Percobaan | Hasil | Status |
|---|---|---|
| Refiner | +0.0177 hold-out | adopsi |
| Near-limb drop t=0.88 | +0.0065 OOF | adopsi |
| Instance scorer | +0.0144 OOF | adopsi |
| EMA checkpoint (ensemble) | −0.010 hold-out; LB probe 0.36 (sama) | tidak |
| Consensus target (5 fold) | tidak lebih baik | tidak |
| Refiner gen-2 (Lovasz, EffNet-B3, crop TTA) | tidak ada gain (refiner jenuh) | tidak |
| Spine multi-task stage-A | +0.0074 tanpa refiner, +0.0016 dengan refiner | tidak |
| Hard-negative refiner | +0.0006 | noise |
| Ensemble + Mask R-CNN publik | +0.0028 | tidak |
| Filter "didukung ≥2 pass" (klasifier) | +0.0024 OOS (oracle +0.0132) | marginal |
| Filter `std_views` | +0.0020 | marginal |
| Re-tune threshold single-pass (300 konfigurasi) | +0.0003 | noise |
| Loose candidates + scorer | +0.0040 OOF (<margin +0.005), −0.0028 hold-out | tidak |
| Fitur refiner untuk scorer | −0.0005 OOF | tidak |
| Per-instance mask-variant selection | −0.0013 OOF (oracle +0.0223 tak terkejar) | tidak |
| Multi-scale TTA, single model OOF (0.75/1.0/1.25) | **+0.0117** (CI95 +0.0067..+0.0169) | positif |
| Multi-scale TTA pada ensemble, hold-out | −0.0114 (CI95 −0.030..+0.005) | gate gagal, test tidak dijalankan |
| Dilasi/erosi mask ±1px (probe LB) | 0.31 / 0.32 (cocok prediksi null) | konvensi label test = train |
| Resampling patch per epoch (1.760 patch unik -> baru tiap epoch), 5 fold, 557 gambar | **-0.0202** (CI95 -0.027..-0.014); semua fold turun; FP +42% | tidak (gate gagal) |
| Pretraining self-supervised GONG H-alpha | belum dilakukan; perkiraan 3–6 GPU-jam + fine-tuning (report.md:99) | di-skip (peluang gain ≥0.01 <20%) |

Pelajaran statistik [fakta]: dengan hold-out 50 gambar, SE paired ≈0.01, sehingga banyak hasil nol bersifat **tidak konklusif**, bukan negatif. Satu model tunggal OOF = 0.388 (rentang fold 0.372–0.407); ensemble menambah ~+0.026.

## 4. Diagnosis: kenapa jauh dari top

### 4.1 Gap tak terjelaskan OOF/hold-out → LB (≈0.04–0.05)
- Expected recipe ≈ 0.41 (OOF 0.388 + ensemble ≈0.026), LB 0.37. [fakta]
- Sudah dieliminasi: campuran tahun/situs observatorium (re-weight −0.002/−0.000), jumlah prediksi per gambar (7.5 test vs 8.4 hold-out vs GT 7.7), offset boundary ±1px (probe LB), single-pass vs multi-pass (OOF +0.010 n.s.). [fakta]
- Petunjuk lemah: mask prediksi 14% lebih besar di test (median 1,320 vs 1,158 px); test punya 14% gambar 2019–2022. [fakta]
- **Hasil A1 (14 Okt, `analysis/scripts/analyze_shift_a1.py`)** [fakta]: tidak ada pergeseran yang terlihat pipeline (AUC test-vs-OOF pada fitur geometri/citra 0.509, dalam rentang permutasi 0.44-0.56). Pembobotan OOF ke distribusi test menggeser PQ 0.4088 -> 0.4142 (menjauh dari LB). Gap tidak dijelaskan oleh kovariat yang teramati; sisanya adalah hal yang tak terlihat dari prediksi (gaya/jumlah pass anotator di label test, derau skor LB publik). Rincian: report.md §4.
- Hipotesis tersisa [inferensi]: distribusi anotator/gaya label berbeda pada test; jumlah pass anotator berbeda; sebagian gambar test di luar domain train. Hanya probe LB yang bisa menguji.

### 4.2 Plafon label
- Anotator vs anotator: PQ **0.340**, IoU matched 0.634; hanya 0.14% pasangan mencapai IoU 0.85. Pipeline (0.4427) sudah melampaui kesepakatan pasangan karena memprediksi rata-rata. [fakta]
- 54% error = near-miss (overlap tapi IoU≤0.5). Oracle membuang instance di bawah break-even akan +0.074; scorer menangkap +0.014. [fakta]
- Ini berarti ruang perbaikan nyata ada di **seleksi objek** dan **boundary konsensus**, bukan di refinement tambahan.

### 4.3 Stage-A adalah bottleneck [inferensi dari fakta]
Fitur refiner, loose candidates, dan variant selection semuanya nol sementara multi-scale pada single model memberi +0.0117. Artinya informasi pembeda TP/FP tidak ada di peta probabilitas stage-A saat ini; harus datang dari model yang berbeda/lebih baik atau dari konteks lebih luas.

### 4.4 Struktur papan peringkat
- 730 tim (27 Sep). 52 tim di 0.49–0.62 = kelompok invalid. 0.41 (2), 0.40 (11), 0.39 (21), 0.37 kita dibagi 67 tim. [fakta]
- Organizer: skor semacam itu "tidak dianggap valid"; PQ>0.35 "of great value"; evaluasi final **bukan** berdasarkan skor LB, menggabungkan kriteria kuantitatif dan kualitatif (halaman Evaluation). [fakta]
- Konsekuensi [inferensi]: LB hanya filter; peluang Top 3 sebenarnya ada di penilaian akhir, dan di sana kualitas laporan + reproducibility + legitimasi berbobot.

## 5. Strategi

### Track A — Kuantitatif (target realistis 0.385–0.41)

Aturan umum: semua keputusan di **OOF 657 gambar, paired bootstrap per gambar**. Hold-out 50 gambar hanya sebagai sanity gate (tidak boleh turun >0.005). Tulis kriteria **sebelum** run (preregistration gaya repo ini). Satu submission LB per perubahan besar; jangan menyetel ke LB.

**A1. Diagnosa gap LB (CPU, murah — kerjakan pertama).**
- Bandingkan distribusi test vs OOF: area instance, jumlah instance per gambar, panjang spine, posisi radial, kontras lokal, tahun/situs per gambar. Cari subset test yang "tidak seperti" train (nearest-neighbour pada fitur gambar/prediksi).
- Hitung PQ OOF yang diboboti ke distribusi test menggunakan fitur tambahan di atas (bukan hanya tahun/situs).
- Keluaran: apakah gap bisa dikaitkan ke subset tertentu. Jika ya, lever training/threshold khusus subset itu.

**A2-0. Resampling patch per epoch (E1; lever termurah, terlewat di versi awal dokumen ini).** report.md:64 (ditemukan 30 Sep, belum diuji) dan kode (`training/filament_unet_ensemble.py:210`, generator di-seed hanya oleh indeks sampel; DataLoader tanpa shuffle) menunjukkan 30 epoch hanya melihat **1.760 patch unik** (52.800 draw). Skrip siap: `training/filament_resample_stagea.py` (fail-fast, stop-on-error, checkpoint, time budget) + `analysis/scripts/analyze_resample.py`. Gate terkunci di skrip: gain gabungan 5 fold dengan refiner >= +0.005 dengan CI95 > 0 terhadap **baseline EMA ckpt_selection (0.3998 pada 557 gambar)**, lalu hold-out ensemble tidak lebih rendah dari ensemble EMA. Biaya ~25-35 menit T4 per fold, ~3 jam + skoring. Jangan pilot 1 fold (pilot +0.0144 menjadi +0.0023 pada 5 fold, report.md:40-42). Gain diharapkan menyusut di ensemble (EMA +0.0144 -> +0.0032; spine +0.0074 -> +0.0016) [inferensi agen tinjauan, belum diverifikasi]. Baseline untuk semua varian A2 = EMA ckpt_selection, bukan "0.388".
**HASIL (14 Okt): DITOLAK.** 5 fold, 557 gambar: EMA 0.3998 -> 0.3796 (-0.0202, CI95 -0.0269..-0.0136); last 0.3977 -> 0.3783; tanpa refiner -0.0337; semua fold turun. TP naik (3.934 -> 4.166) tetapi FP naik jauh lebih banyak (2.909 -> 4.121): ambang kandidat, refiner, dan scorer yang disetel pada model lama tidak cocok dengan peta model baru. Tidak diuji ulang dengan ambang baru (butuh inferensi GPU lagi; defisit -0.02 membuat gain bersih tidak mungkin). Biaya: 2.7 jam T4. Rincian: report.md §4.

**A2. Stage-A lebih kuat (GPU, inti track).**
- Konteks lebih besar: patch 768/1024 (sekarang 512) dengan batch/akumulasi gradient menyesuaikan. Dasar: multi-scale gain menunjukkan sensitivitas skala.
- Diversitas encoder: ConvNeXt/EfficientNet-B3–B5/SegFormer (smp mendukung). Mix R34+R50 sebelumnya 0.34 LB pada proxy lama — ulangi dengan protokol OOF, bukan proxy.
- Lebih banyak seed (10–15 model); estimasi gain ensemble marginal +0.003–0.006 [inferensi].
- Evaluasi: OOF single-model dulu (baseline 0.388), lalu ensemble penuh. Gate: gain OOF ≥ +0.005 dengan CI95 > 0.

**A3. Re-tune kandidat pada peta multi-scale (CPU setelah A2).**
- Threshold kandidat saat ini disetel untuk peta single-scale; hold-out ensemble kehilangan recall pada 3-scale (TP 394→356). Re-tune `prob`, area, mean prob dan refit scorer pada peta multi-scale OOF single-model (satu-satunya yang punya 657 gambar). Fit scorer ulang karena kalibrasi probabilitas bergeser.
**Peringatan:** gain multi-scale tidak berpindah ke ensemble (hold-out -0.0114); kemungkinan ensemble 20 prediksi sudah membuang varians yang sama [inferensi]. Nilai harapan A3 untuk ensemble rendah; jalankan hanya jika A2 menghasilkan model tunggal baru yang butuh ambang baru. Hal yang sama berlaku untuk "lebih banyak seed" di A2.

**A4. Target konsensus pada subset multi-pass (GPU, satu percobaan terbatas).**
- Simulasi leave-one-pass-out: objek yang ditandai ≥2 pass dengan bentuk union = +0.0145 (CI +0.006..+0.023). Pilot fold-1 sebelumnya "tidak lebih baik". Coba hanya pada 296 gambar multi-pass sebagai fine-tune, dievaluasi OOF; jika tidak lulus gate, hentikan.

**A5. Hal yang jangan diulang** (sudah mati, lihat §3): resampling patch per epoch (A2-0), refiner gen-2, spine multi-task, variant selection, loose candidates tanpa re-tune multi-scale, EMA ensemble, fitur refiner untuk scorer, Mask R-CNN publik.

**A6. Prioritas rendah (bukan mustahil).** Pretraining self-supervised pada GONG H-alpha tak berlabel diizinkan organizer (forum 737215, 744047; gambar tak berlabel tidak boleh mencakup test atau ±1 jam dari timestamp test, forum 748045; wajib diungkap di laporan). Biaya menurut report.md:99-100: sekitar **3-6 GPU-jam** plus fine-tuning (koreksi: versi awal dokumen ini salah menulis "GPU-hari"). Report menilai peluang gain ≥0.01 di bawah 20% dan tidak ada bukti publik bahwa ini mengalahkan inisialisasi ImageNet untuk H-alpha. Kerjakan hanya setelah E1 (A2-0) selesai dan masih ada kuota.

**Anggaran GPU A2 [inferensi dari ~35 menit/fold pada patch 512]:** satu varian dengan OOF 5 fold ~3 jam T4 (patch 512) sampai ~3-7 jam (patch 768). Kuota hanya cukup untuk 2-3 varian. Jalankan satu per satu dan mintalah izin per kernel.

### Track B — Kualitatif (di sini peluang Top 3 paling nyata)

1. **Klaim tunggal yang tajam** di abstrak: skor dibatasi oleh kesepakatan anotator (PQ 0.34); aturan break-even PQ/2 menghasilkan scorer yang +0.014 OOF tanpa tuning; boundary berada di optimum tajam (±1px = −0.04..−0.05).
2. **Reproducibility (BERSYARAT, keputusan pemilik repo)**: kode sudah publik (Apache-2.0); bobot sengaja ditahan sejak 2026-09-23 (README "Weights availability": tidak dipublikasikan sampai mendekati 15 Nov agar kompetitor tidak membangun di atas checkpoint ini). Hasil cek aturan (§7b): repo harus publik saat kompetisi tutup dan reproducible tanpa file tambahan; bobot tidak disebut eksplisit, tetapi tanpa bobot yang terjangkau penilaian tidak bisa dilakukan. Rencana: tahan sampai mendekati 15 Nov (setelah submission final), lalu publikasikan dengan `requirements.txt` terpin dan tautan di laporan; simpan `SHA256SUMS.txt`. Opsi "publikasikan hanya jika menang" berarti tidak dinilai. Memublikasikan adalah tindakan ke luar yang tidak bisa ditarik kembali; butuh persetujuan eksplisit pemilik.
3. **Laporan 4 halaman template ACM** (`report/overleaf/`): masukkan skor LB final, hasil A1–A4 (termasuk negatif), dan satu tabel "apa yang membatasi" (annotator ceiling, error decomposition 54% near-miss).
4. **Pertahankan sikap legitimasi**: §7–§8 report (tidak mengejar ujung atas LB, tidak memakai anotasi di luar kompetisi, tidak memakai isi notebook pre-computed). Ini nilai plus pada penilaian kualitatif.
5. Figur kualitatif sudah ada (`report/figures/fig_morph_*.png`); tambahkan kegagalan-kasus terbaik dan satu demonstrasi scorer (instance yang dibuang vs dipertahankan).

## 6. Batasan dan aturan keras

- **Jangan** memakai rilis publik MAGFiLO yang menurut forum memuat anotasi gambar test (topik 728062, 737177, 741892), file submission pihak ketiga, maupun isi notebook pre-computed. Organizer menyatakan submission semacam itu didiskualifikasi (topik 729396). [fakta]
- Pseudo-labeling pada 180 gambar test: **cek aturan** sebelum dipakai; default = jangan.
- Gate preregistered: tulis hipotesis, metrik, dan ambang keputusan di commit sebelum menjalankan; catat hasil negatif di `report/report.md` §4.
- Satu ide per percobaan; jangan menumpuk perubahan sebelum masing-masing lulus gate.
- Jangan menghapus artefak OOF/hasil di `analysis/results/` (dipakai semua analisis).

## 7. Peta file

```
filament_pipeline.py      inferensi (model, TTA, kandidat, refiner, near-limb, RLE, validasi format)
filament_scorer.py        fitur instance + aturan keep/drop; models/ berisi LightGBM + threshold
filament_metric.py        PQ resmi per pass anotator; hold-out tetap (seed 1234)
pipeline.ipynb            notebook end-to-end -> submission.csv
training/                 filament_unet_ensemble.py, filament_refiner.py, *_gen2/spine/consensus/ckpt_selection/resample_stagea
analysis/scripts/         filament_*_cpu.py / *_gpu.py (kernel), analyze_*.py (analisis lokal, baca analysis/results/)
analysis/results/         keluaran studi (OOF counts, instance, dst.)
report/report.md          laporan teknis panjang (semua temuan)
report/overleaf/          laporan 4 halaman ACM + zip Overleaf
tests/                    pytest: metrik, kandidat, near-limb, submission, scorer
```

Perintah uji: `python -m pytest tests`.

## 7b. Aturan organizer (dibaca dari halaman kompetisi 14 Okt; sumber: ringkasan agen, belum saya buka sendiri)

- **Penilaian final = rubrik, bukan LB**: kuantitatif 70% (mean Dice, PQ, distribusi Dice/IoU, distribusi one-to-many/many-to-one) + kualitatif 30% (deskripsi pipeline, morfologi prediksi, kualitas kode). Evaluasi "contingent upon the availability of the source code". Kita hanya mengoptimalkan PQ; Dice dan statistik one-to-many/many-to-one ikut dinilai.
- **Publikasi wajib untuk dinilai**: laporan 4 halaman + repo Git publik dengan `requirements.txt` berversi + notebook pipeline; "should not require any additional file to reproduce". Repo harus publik begitu kompetisi tutup (15 Nov 2026 23:59 CDT) dan tetap publik sampai pemenang diumumkan (30 Nov). Bobot tidak disebut eksplisit, tetapi prediksi harus reproducible tanpa file tambahan, jadi bobot praktis harus terjangkau. Rencana README (publikasi mendekati 15 Nov) konsisten; "publikasi hanya jika menang" berarti tidak dinilai.
- **LB**: menampilkan ~50% gambar test; 5 submission/hari; pilih sampai 2 final submission, keduanya harus reproducible.
- **Hadiah**: pool maks $3.000 untuk top-3 menurut skor rubrik; syarat: kode training+inference, laporan, lisensi OSI, bisa menjelaskan solusi.
- **Data/pretraining**: data eksternal publik boleh; SSL pada GONG tak berlabel boleh (jangan mencakup test atau ±1 jam dari timestamp test; ungkapkan); pseudo-label/TTT pada gambar test tidak disebut eksplisit dan "extra information at inference" dilarang: **default jangan**.

## 7c. Metrik rubrik selain PQ (dihitung 14 Okt; `analysis/scripts/analyze_rubric.py`, `analysis/results/rubric_summary.json`)

Definisi dari notebook self-evaluation organizer (matriks IoU/Dice berpasangan GT x prediksi per anotator-gambar; distribusi hanya pasangan skor > 0; one-to-many / many-to-one = jumlah mitra dengan IoU > 0). Konfigurasi `torchmetrics DiceScore` **tidak dinyatakan**; kami melaporkan Dice foreground (gabungan mask) per gambar, plus varian macro dan micro.

| | OOF 657 | hold-out 50 |
|---|---|---|
| PQ (near-limb + scorer, tau 0.1972) | 0.4089 | 0.4427 |
| Mean Dice per gambar | 0.624 | 0.664 |
| Mean Dice tanpa scorer | 0.6326 | 0.6758 |
| Selisih Dice karena scorer | -0.0084 (CI95 -0.014..-0.0035) | -0.0123 (CI95 -0.021..-0.005) |
| GT tanpa prediksi (1:0) | 26.8% | 24.4% |
| Prediksi tanpa GT (0:1) | 12.3% | 10.0% |
| Prediksi menyentuh >= 2 GT | 2.4% | 1.9% |
| GT terpecah menjadi >= 2 prediksi | 3.4% | 3.7% |

- **Temuan**: scorer menaikkan PQ (+0.0144) tetapi menurunkan Dice (-0.0084) karena membuang 16% instance. Sapuan ambang di OOF (PQ / Dice): 0.10: 0.4022 / 0.6320; 0.15: 0.4063 / 0.6313; **0.197: 0.4088 / 0.6241**; 0.25: 0.4069 / 0.6123. Dice jatuh di antara ambang 0.15 dan 0.25.
- Bobot Dice vs PQ dalam rubrik tidak dipublikasikan, sehingga ambang yang cocok untuk evaluasi final tidak bisa diturunkan; 0.197 adalah optimum PQ. Dengan bobot sama, ambang ~0.15 lebih baik sekitar +0.005 pada jumlah PQ+Dice (seleksi satu parameter pada OOF yang sama; tanpa konfirmasi held-out).
- Over-merging jarang (2.4%) dan fragmentasi kecil (3.4%); yang dominan adalah miss.
- Dua final submission diperbolehkan, tetapi host menyarankan menggabungkannya menjadi satu strategi (forum 747960; keduanya harus reproducible).
- **Dua varian disiapkan** (`fs.InstanceScorer.load(tau=...)`; `py analysis/scripts/make_scorer_submission.py <dir> [tau]`; belum disubmit):

| | OOF 657 PQ / Dice | hold-out 50 PQ / Dice | instance test dikirim |
|---|---|---|---|
| **rilis** tau 0.1972 | 0.4089 / 0.6241 | 0.4427 / 0.6635 | 1.083 / 1.267 |
| **seimbang** tau 0.15 | 0.4063 / 0.6313 | 0.4418 / 0.6681 | 1.143 / 1.267 |
| selisih (seimbang - rilis), berpasangan | PQ -0.0027 (CI -0.0046..-0.0007); Dice +0.0072 (CI +0.004..+0.011) | PQ -0.0009 (CI -0.009..+0.006); Dice +0.0046 (CI +0.001..+0.008) | |

Ambang 0.15 dipilih pada OOF yang sama (satu parameter, tanpa konfirmasi independen).

## 8. Hal yang HARUS dicek dulu (tidak diketahui)

1. **Deadline** submission final: **15 Nov 2026** (README baris 50; sisa 32 hari per 2026-10-14). Deadline laporan terpisah: **[tidak diketahui]**, cek. Dengan 32 hari dan kuota GPU bersama, A6 tidak mungkin dan A2 dibatasi 2-3 varian.
1a. **Porsi LB publik** dari 180 gambar test dan batas submission harian: **[tidak diketahui]**, cek di halaman kompetisi. Jika porsinya kecil, SE LB publik sebanding dengan hold-out (~0.017) dan skor dibulatkan ke 0.01, sehingga probe LB tidak bisa mendeteksi gain di bawah ~0.01-0.02. Keputusan tetap di OOF; LB hanya sanity check.
1b. **Bobot tidak ada di repo/laptop** (`models/` hanya scorer). Ambil dari output kernel Kaggle `filament-unet-ensemble-5fold` dan `filament-crop-refiner` sebelum Track B-2.
2. **Batas submission harian** dan **sisa kuota GPU Kaggle** (T4; multi-scale OOF memakai ~114 menit dua kernel; ensemble 50 gambar 29 menit).
3. **Posisi LB terkini** dan apakah papan sudah direfresh organizer (kelompok invalid masih ada?).
4. Aturan resmi tentang pseudo-label/test-time training dan batas penggunaan model pre-trained eksternal.
5. Apakah "final submission" dipilih manual (maks N) atau otomatis — pilih berdasarkan OOF + hold-out, bukan LB tertinggi.

## 9. Urutan kerja yang disarankan untuk sesi baru

1. Baca `report/report.md` §3–§5 dan §7; jalankan `python -m pytest tests`.
2. Selesaikan §8 (deadline, kuota).
3. A1 (CPU) → tulis temuan ke `analysis/` + ringkasan di report.
4. Rancang A2 sebagai preregistration (satu varian per kernel, OOF 657), jalankan varian termurah dulu (patch 768 pada satu fold sebagai pilot).
5. A3 setelah ada peta multi-scale; A4 jika sisa waktu.
6. Satu probe LB per lulus gate; catat semua hasil.
7. Track B paralel: cek aturan publikasi bobot (B-2, bersyarat), perbarui laporan ACM, lalu final selection.

## 10. Ekspektasi yang jujur

- Top 3 di LB valid (≥0.40-0.41): peluang sangat rendah (perkiraan satu digit persen [inferensi]; revisi dari 10-15%). Butuh +0.03-0.04, sedangkan gain yang masuk akal berjumlah ~+0.01-0.02 dan belum terbukti pada ensemble; bagian gap yang tak terjelaskan mungkin pergeseran distribusi yang tidak bisa diperbaiki lewat training.
- Skor 0.385–0.395 realistis jika A2+A3 berhasil, menaikkan peringkat valid dari ~96 ke sekitar 30–40 [inferensi].
- Nilai tambah terbesar per jam kerja ada di Track B dan di A1 (menjelaskan gap), karena keduanya bukan sekadar menyetel ke LB dan menarik di penilaian kualitatif.
