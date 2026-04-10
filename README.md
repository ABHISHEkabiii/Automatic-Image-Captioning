# 🖼️ Automatic Image Captioning

> **Deep Learning pipeline for generating natural language descriptions from images — combining CNN + LSTM (classical) with BLIP Transformer (state-of-the-art).**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow)](https://tensorflow.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-BLIP-yellow?logo=huggingface)](https://huggingface.co/Salesforce/blip-image-captioning-large)
[![Dataset](https://img.shields.io/badge/Dataset-Flickr8k-green)](https://academictorrents.com/details/9dea07ba660a722ae1008c4c8afdd303b6910f61)
[![BLEU-1](https://img.shields.io/badge/BLEU--1-0.35-brightgreen)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## 📌 Table of Contents

- [Overview](#overview)
- [Project A — CNN + LSTM (VGG16 + RNN)](#project-a--cnn--lstm-vgg16--rnn)
- [Project B — BLIP Transformer (Advanced)](#project-b--blip-transformer-advanced)
- [Architecture Comparison](#architecture-comparison)
- [Results](#results)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quickstart](#quickstart)
- [Authors](#authors)

---

## Overview

This repository contains **two complementary image captioning systems** built as part of academic coursework at **VIT Vellore (Integrated M.Sc. — Computational Statistics & Data Analytics)**:

| | Project A | Project B |
|---|---|---|
| **Architecture** | VGG16 (CNN) + LSTM (RNN) | BLIP (Vision-Language Transformer) |
| **Approach** | Classical encoder-decoder | Bootstrapped pre-training |
| **Dataset** | Flickr8k (8,000 images) | Any user-uploaded image |
| **Inference** | Beam search + top-k sampling | Conditional & unconditional |
| **BLEU-1** | **0.35** | N/A (zero-shot) |
| **Framework** | TensorFlow / Keras | PyTorch + HuggingFace |

Both systems take an image as input and output a human-readable natural language description — bridging **computer vision** and **NLP** in a single pipeline.

---

## Project A — CNN + LSTM (VGG16 + RNN)

### How It Works

The pipeline follows a classic **encoder-decoder** architecture:

```
Input Image
    │
    ▼
┌─────────────────────────────────────────┐
│  ENCODER: VGG16 (pre-trained, ImageNet) │
│  Input: 224 × 224 × 3                  │
│  Output: 4096-dimensional feature vector│
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  DECODER: LSTM (Language Model)         │
│  Word Embedding → LSTM → Dense(vocab)  │
│  Generates one word at a time           │
│  Stops at <endseq> token                │
└──────────────────┬──────────────────────┘
                   │
                   ▼
         Generated Caption 🗒️
```

### Dataset — Flickr8k

- **8,091 images** with **5 human-annotated captions each** (40,455 total)
- Split: 6,000 train / 1,000 validation / 1,000 test
- Source: [Flickr8k via Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k)

**Sample images and captions from Flickr8k:**

> *"beige puppy walks across the floor"*
> *"blond girl in brown shirt with black pen up her nose"*
> *"little girl smiles as she wears white bowl on the top of her head"*

### Data Preprocessing

Three cleaning functions applied to all captions:

```python
def clean_captions(mapping):
    for key, captions in mapping.items():
        for i, caption in enumerate(captions):
            caption = caption.lower()
            caption = re.sub(r'[^a-z\s]', '', caption)   # remove punctuation & digits
            caption = re.sub(r'\s+', ' ', caption)         # remove extra spaces
            caption = 'startseq ' + caption + ' endseq'   # add sequence tokens
            captions[i] = caption
```

**Vocabulary after cleaning:** ~8,000 unique tokens (reduced by ~200 from raw).

### Image Feature Extraction (VGG16)

```python
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model

# Load VGG16 without the classification head
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)

# Extract 4096-dim features for each image
def extract_features(directory):
    features = {}
    for img_name in os.listdir(directory):
        img = load_img(path, target_size=(224, 224))
        img = img_to_array(img)
        img = preprocess_input(img)
        feature = model.predict(img, verbose=0)
        features[img_name] = feature
    return features
```

Features were visualised using **PCA** (4096 → 2D) and **k-means clustering** (k=4) to verify semantic grouping — images in the same cluster visually resemble each other, confirming the representations are meaningful.

### Model Architecture (Encoder-Decoder)

```python
# Image feature input
inputs1 = Input(shape=(4096,))
fe1 = Dropout(0.4)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)

# Caption sequence input
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.4)(se1)
se3 = LSTM(256)(se2)

# Merge + decode
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs  = Dense(vocab_size, activation='softmax')(decoder2)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam')
```

| Component | Detail |
|-----------|--------|
| Image encoder | VGG16 → 4096 → Dense(256) |
| Text encoder | Embedding(256) → LSTM(256) |
| Merge | Element-wise add |
| Decoder | Dense(256, relu) → Dense(vocab, softmax) |
| Optimizer | Adam |
| Loss | Categorical cross-entropy |
| Epochs | 20 |
| Batch size | 32 |

### Optimizer Study

Four optimizers were compared — **Adam outperformed all others**:

| Optimizer | Validation Loss | Notes |
|-----------|----------------|-------|
| **Adam** | **Lowest** | Momentum + adaptive LR — best convergence |
| RMSProp | Moderate | Adaptive LR only |
| SGD + Momentum | Higher | Stable but slow |
| Vanilla SGD | Highest | Prone to oscillation |

Adam combines **RMSProp** (adaptive learning rate) and **momentum**, making it ideal for sparse gradients in NLP tasks.

### Inference — Beam Search

```python
def generate_caption(model, tokenizer, photo, max_length, beam_width=5):
    # Greedy as beam_width=1; beam search explores top-k paths
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None or word == 'endseq':
            break
        in_text += ' ' + word
    return in_text
```

---

## Project B — BLIP Transformer (Advanced)

The second system uses **Salesforce BLIP** (Bootstrapping Language-Image Pre-training), a state-of-the-art vision-language model that drastically outperforms classical CNN+LSTM approaches — no training required.

### Architecture

```
Input Image (any resolution)
        │
        ▼
┌────────────────────────────────────────────────┐
│  BLIP Vision Encoder (ViT-Large)               │
│  Patch embeddings + cross-attention            │
└──────────────────────┬─────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        ▼                             ▼
 Unconditional                  Conditional
 (free generation)          (prompted: "a photo of")
        │                             │
        ▼                             ▼
┌───────────────┐           ┌──────────────────────┐
│  BERT Decoder │           │  BERT Decoder         │
│  (text-only)  │           │  (text + image cross) │
└───────────────┘           └──────────────────────┘
        │                             │
        ▼                             ▼
"there are four dogs         "a photo of a group of dogs
running together in          running across a lush
a line on the grass"         green field"
```

### Implementation

```python
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-large"
).to(device)

def caption_image(img_path):
    img = Image.open(img_path).convert('RGB')

    # Unconditional caption
    inputs1 = processor(img, return_tensors='pt').to(device)
    cap1 = processor.decode(
        model.generate(**inputs1, max_new_tokens=50, num_beams=5)[0],
        skip_special_tokens=True
    )

    # Conditional caption (prompted)
    inputs2 = processor(img, text="a photo of", return_tensors='pt').to(device)
    cap2 = processor.decode(
        model.generate(**inputs2, max_new_tokens=50, num_beams=5)[0],
        skip_special_tokens=True
    )

    return cap1, cap2
```

### Sample Output

**Input:** Dogs running on grass

| Mode | Caption |
|------|---------|
| Unconditional | *"there are four dogs running together in a line on the grass"* |
| Conditional | *"a photo of a group of dogs running across a lush green field"* |

> **Hardware used:** Google Colab T4 GPU · Model size: 1.88 GB · Load time: ~16s

---

## Architecture Comparison

| Feature | Project A (CNN+LSTM) | Project B (BLIP) |
|---------|---------------------|------------------|
| **Model type** | Custom encoder-decoder | Pre-trained transformer |
| **Parameters** | ~10M (trainable) | ~470M (frozen) |
| **Training** | 20 epochs on Flickr8k | Zero-shot (pre-trained) |
| **GPU time** | ~2–3 hrs (T4) | ~16s load, <1s inference |
| **BLEU-1** | 0.35 | N/A (zero-shot) |
| **Flexibility** | Domain-specific fine-tuning | General-purpose |
| **Captioning mode** | Single output | Conditional + unconditional |
| **Framework** | TensorFlow/Keras | PyTorch + HuggingFace |
| **Best for** | Learning fundamentals | Production deployment |

---

## Results

### Project A — BLEU Scores

Evaluated on Flickr8k test set (1,000 images, 5 reference captions each):

| Metric | Score |
|--------|-------|
| **BLEU-1** | **0.35** |
| BLEU-2 | 0.21 |
| BLEU-3 | 0.14 |
| BLEU-4 | 0.09 |

**BLEU** (Bilingual Evaluation Understudy) measures n-gram overlap between generated and reference captions. Score of 1.0 = exact match; 0.0 = no overlap.

**Examples with BLEU ≥ 0.70:**
> Predicted: *"a dog runs through the grass"*
> Reference: *"a brown dog is running through the grass"*

**Examples with BLEU 0.30–0.70:**
> Predicted: *"a man in a red shirt is riding a bike"*
> Reference: *"a cyclist wearing red rides down a hill"*

### Training Dynamics

- **Epochs 1–10:** Both train and val loss decrease rapidly
- **Epochs 10–15:** Val loss plateaus; model checkpoint saved at minimum
- **Epochs 15–20:** Sign of mild overfitting — early stopping applied

### Project B — Qualitative Results

BLIP produces significantly richer, more contextually accurate captions due to pre-training on hundreds of millions of image-text pairs from the web. Conditional captioning (with a text prompt) tends to produce more descriptive, fluent output.

---

## Tech Stack

### Project A
```
Python 3.8+        — Core language
TensorFlow 2.x     — Model training
Keras              — High-level API
VGG16              — Pre-trained CNN feature extractor
NLTK               — BLEU score evaluation
NumPy / Pandas     — Data handling
Matplotlib         — Visualisation
scikit-learn       — PCA, k-means clustering
```

### Project B
```
Python 3.10+           — Core language
PyTorch 2.x            — Deep learning backend
HuggingFace Transformers — BLIP model + processor
Pillow                 — Image loading
Google Colab (T4 GPU)  — Execution environment
Matplotlib             — Output visualisation
```

---

## Project Structure

```
automatic-image-captioning/
│
├── 📂 project_a_cnn_lstm/
│   ├── Automatic_Image_Captioning_v4.ipynb   ← Main notebook
│   ├── feature_extraction.py                 ← VGG16 feature pipeline
│   ├── data_preprocessing.py                 ← Caption cleaning + tokenisation
│   ├── model.py                              ← Encoder-decoder definition
│   ├── train.py                              ← Training loop + checkpoints
│   ├── evaluate.py                           ← BLEU score computation
│   └── inference.py                          ← Beam search caption generation
│
├── 📂 project_b_blip/
│   └── Image_Captioning_with_BLIP_Colab.ipynb  ← BLIP demo notebook
│
├── 📂 data/
│   └── flickr8k/                             ← Place dataset here
│       ├── Images/
│       └── captions.txt
│
├── 📄 AUTOMATIC_IMAGE_CAPTIONING.pdf         ← Research paper (VIT Vellore)
├── 📄 README.md
└── 📄 requirements.txt
```

---

## Quickstart

### Project A — CNN + LSTM

```bash
# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/automatic-image-captioning
cd automatic-image-captioning
pip install -r requirements.txt

# 2. Download Flickr8k dataset
# Place in data/flickr8k/Images/ and data/flickr8k/captions.txt

# 3. Run the notebook
jupyter notebook project_a_cnn_lstm/Automatic_Image_Captioning_v4.ipynb
```

**requirements.txt (Project A)**
```
tensorflow>=2.10
numpy
pandas
matplotlib
nltk
scikit-learn
pillow
tqdm
```

### Project B — BLIP (Google Colab)

Open directly in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19PTO84-InVnbuPbI6gGhKjwGKYm93Cot)

```bash
# Or run locally
pip install transformers torch torchvision pillow

python - <<'EOF'
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-large"
).to(device)

img = Image.open("your_image.jpg").convert("RGB")
inputs = processor(img, return_tensors="pt").to(device)
caption = processor.decode(
    model.generate(**inputs, max_new_tokens=50, num_beams=5)[0],
    skip_special_tokens=True
)
print(caption)
EOF
```

---

## Authors

| Name | Role | Contact |
|------|------|---------|
| **Abhishek** | Model architecture, training, evaluation , BLIP integration| abhishek.2020@vitstudent.ac.in |

**Institution:** School of Advanced Sciences, Vellore Institute of Technology (VIT), Vellore

---

## References

1. Rashtchian et al. *Collecting Image Annotations Using Amazon's Mechanical Turk.* NAACL HLT 2010.
2. Simonyan & Zisserman. *Very Deep Convolutional Networks for Large-Scale Image Recognition.* ICLR 2015.
3. Xu et al. *Neural Image Caption Generation with Visual Attention.* ICML 2015.
4. Li et al. *BLIP: Bootstrapping Language-Image Pre-training.* ICML 2022.
5. Papineni et al. *BLEU: A Method for Automatic Evaluation of Machine Translation.* ACL 2002.
6. He et al. *Deep Residual Learning for Image Recognition.* CVPR 2016.

---

<div align="center">
  <sub>Built by Abhishek at VIT Vellore · School of Advanced Sciences · 2024–2025</sub>
</div>
