# 🖼️ Automatic Image Captioning
### CNN (VGG16) + LSTM Encoder-Decoder | Flickr8k | Beam Search & Top-k Sampling

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.0-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?logo=keras&logoColor=white)](https://keras.io)
[![Dataset](https://img.shields.io/badge/Dataset-Flickr8k-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/adityajn105/flickr8k)
[![Colab](https://img.shields.io/badge/Run%20on-Google%20Colab-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> A deep learning system that **automatically generates natural language descriptions for images** by combining Convolutional Neural Networks for visual feature extraction with Long Short-Term Memory networks for sequential caption generation.

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Architecture](#-architecture)
- [Dataset](#-dataset)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Quickstart](#-quickstart)
- [Implementation Details](#-implementation-details)
- [Evaluation](#-evaluation)
- [Future Work](#-future-work)

---

## 🔍 Overview

Automated image captioning sits at the intersection of **Computer Vision** and **Natural Language Processing**. This project implements an encoder-decoder architecture that:

- Encodes images into a 4096-dimensional feature vector using **pre-trained VGG16**
- Decodes features into captions word-by-word using an **LSTM language model**
- Uses **Beam Search** (width=5) and **Top-k Sampling** (k=5, temp=0.7) for diverse, high-quality outputs
- Trains on **8,091 images** from Flickr8k with 5 reference captions each

**Real output from the model:**
```
Input Image   →  two dogs are playing in the grass with tennis ball in its mouth  (Beam Search)
              →  dog is walking through field of grass                             (Top-k Sampling)
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ENCODER  (Image Branch)                      │
│   Input Image (224×224×3)  →  VGG16 (pretrained, frozen)       │
│   → FC Layer [-2]  →  4096-d feature vector                     │
│   → Dropout(0.5)   →  Dense(512, ReLU)                          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │    ADD (Merge Layer)    │
              └────────────┬────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────────┐
│                   DECODER  (Language Branch)                     │
│   Partial Caption  →  Embedding(vocab_size, 512, mask_zero=True) │
│   → Dropout(0.5)   →  LSTM(512 units)                           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │  Dense(512, ReLU)       │
              │  Dense(vocab_size, Softmax) │
              │  → Predicted Next Word  │
              └─────────────────────────┘
```

> The model is trained end-to-end with **Adam optimizer** and **categorical cross-entropy loss**, with `EarlyStopping(patience=3)` and `ModelCheckpoint` saving the best weights.

---

## 📦 Dataset

**Flickr8k** — 8,091 images with 5 human-annotated captions each (40,455 total captions).

| Split | Images | Captions |
|-------|--------|----------|
| Train (90%) | 7,281 | 36,405 |
| Test  (10%) | 810   | 4,050  |

**Caption preprocessing pipeline:**
```
Raw caption  →  Lowercase  →  Remove punctuation & digits
             →  Remove single-character tokens
             →  Add <startseq> / <endseq> tokens
```

**Sample cleaned captions for one image:**
```
startseq child in pink dress is climbing up set of stairs in an entry way endseq
startseq girl going into wooden building endseq
startseq little girl climbing into wooden playhouse endseq
startseq little girl climbing the stairs to her playhouse endseq
startseq little girl in pink dress going into wooden cabin endseq
```

**Vocabulary stats after cleaning:**
- Vocab size: **8,347 unique tokens**
- Max caption length: **34 tokens**

---

## 📊 Results

### BLEU Score Evaluation (500 test images)

| Metric | Score |
|--------|-------|
| BLEU-1 | **0.3500** |
| BLEU-2 | **0.1471** |
| BLEU-4 | **0.0401** |

> BLEU scores are computed using `nltk.translate.bleu_score.corpus_bleu` against 5 reference captions per image.

### Caption Examples

| Method | Generated Caption |
|--------|------------------|
| 🔵 Beam Search | `two dogs are playing in the grass with tennis ball in its mouth` |
| 🟠 Top-k Sampling | `dog is walking through field of grass` |

### Decoding Strategies Compared

| Strategy | How it works | Output style |
|----------|-------------|--------------|
| **Greedy** | Always picks top-1 word | Repetitive, safe |
| **Beam Search** (width=5) | Tracks top-5 paths simultaneously, length-normalized | Accurate, coherent |
| **Top-k Sampling** (k=5, temp=0.7) | Samples from top-5 with temperature scaling | Natural, diverse |

---

## 📁 Project Structure

```
automatic-image-captioning/
│
├── main_code.ipynb              # Full Colab notebook (13 cells)
├── README.md
│
├── assets/                      # Output images for README
│   ├── architecture.png
│   ├── predictions.png
│   └── training_loss.png
│
├── flickr8k/                    # Downloaded via Kaggle API
│   ├── Images/                  # 8,091 .jpg files
│   └── captions.txt             # image, caption CSV
│
└── saved/
    ├── best_model.keras         # Best checkpoint (saved by ModelCheckpoint)
    ├── features_vgg16.pkl       # Cached VGG16 features (4096-d per image)
    └── tokenizer.pkl            # Fitted Keras Tokenizer
```

---

## ⚡ Quickstart

### Run on Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com/) → Upload `main_code.ipynb`
2. Set runtime: `Runtime → Change runtime type → T4 GPU`
3. Get your Kaggle API token:
   - Go to → https://www.kaggle.com/settings → **API** → **Create New Token**
   - This downloads `kaggle.json` to your PC
4. Run **Cell 2** — a file picker appears, upload `kaggle.json`
5. Run all remaining cells top to bottom ✅

### Local Installation

```bash
git clone https://github.com/yourusername/automatic-image-captioning.git
cd automatic-image-captioning
pip install tensorflow pillow tqdm nltk kaggle matplotlib
kaggle datasets download -d adityajn105/flickr8k --unzip -p flickr8k/
jupyter notebook main_code.ipynb
```

---

## 🔧 Implementation Details

### VGG16 Feature Extraction
```python
base  = VGG16()
model = Model(inputs=base.inputs, outputs=base.layers[-2].output)
# outputs 4096-d vector per image, saved to features_vgg16.pkl
```
Features are extracted once and cached — subsequent runs load from `.pkl` instantly.

### Model Architecture
```python
# embed_dim=512, lstm_units=512
img_input = Input(shape=(4096,))
img_dense = Dense(512, activation='relu')(Dropout(0.5)(img_input))

seq_input = Input(shape=(max_len,))
seq_lstm  = LSTM(512)(Dropout(0.5)(Embedding(vocab_size, 512, mask_zero=True)(seq_input)))

decoder = Dense(512, activation='relu')(Add()([img_dense, seq_lstm]))
output  = Dense(vocab_size, activation='softmax')(decoder)
```

### Training Config
```python
Optimizer  : Adam
Loss       : Categorical Crossentropy
Epochs     : 20 (EarlyStopping patience=3)
Batch size : 32
GPU        : NVIDIA T4 (Google Colab)
TF version : 2.19.0
Seed       : 42 (fully reproducible)
```

### Beam Search Decoding
```python
def predict_caption_beam(model, feature, tokenizer, max_len, beam_width=5):
    beams = [(0.0, ['startseq'])]
    for _ in range(max_len):
        candidates = []
        for score, cap in beams:
            pred = model.predict([feature, seq])
            for idx in np.argsort(pred)[-beam_width:]:
                new_score = score - np.log(pred[idx] + 1e-10)
                candidates.append((new_score, cap + [word]))
        beams = sorted(candidates)[:beam_width]
    # length-normalized selection
    return best_caption
```

---

## 📐 Evaluation

**BLEU (Bilingual Evaluation Understudy)** measures n-gram overlap between generated and reference captions:

- **BLEU-1** → unigram precision (individual word matches)
- **BLEU-2** → bigram precision
- **BLEU-4** → 4-gram precision (strictest, standard benchmark)

A score of **0.35 BLEU-1** is consistent with published results for single-layer LSTM models trained on Flickr8k without attention mechanisms.

---

## 🔮 Future Work

- [ ] Add **attention mechanism** (Show, Attend and Tell) for better spatial focus
- [ ] Replace VGG16 with **ResNet50 / EfficientNet** for richer features
- [ ] Train on **Flickr30k or MS-COCO** (larger datasets) for higher BLEU
- [ ] Implement **transformer-based decoder** (GPT-2 / BERT)
- [ ] Deploy as a **Flask/Streamlit web app**
- [ ] Evaluate with **METEOR, CIDEr, ROUGE** metrics

---

## 📚 References

1. Vinyals et al. — *Show and Tell: A Neural Image Caption Generator* (2015)
2. Xu et al. — *Show, Attend and Tell: Neural Image Caption Generation with Visual Attention*, ICML 2015
3. Simonyan & Zisserman — *Very Deep Convolutional Networks for Large-Scale Image Recognition* (VGG16), ICLR 2015
4. Papineni et al. — *BLEU: a Method for Automatic Evaluation of Machine Translation*, IBM 2002
5. Flickr8k Dataset — Rashtchian et al., NAACL HLT 2010

---

## 👤 Author

**Abhishek**  
M.Sc. Computational Statistics & Data Analytics — VIT Vellore  
School of Advanced Sciences

---

*Built with ❤️ using TensorFlow 2.19 · Google Colab T4 GPU · Flickr8k*
