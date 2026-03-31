# English → Hindi Translation with Transformers

Experience the translation model here: [English → Hindi Translator](https://english-hindi-translation-07.streamlit.app/)

---

## Overview

This project implements a **custom Encoder-Decoder Transformer architecture** specifically designed for translating English sentences into Hindi.  
By leveraging **self-attention** and **cross-attention**, the model captures linguistic nuances, preserving **grammatical structure**, **context**, and **semantic meaning** in translations.

---

## Model Specifications

- **Total Parameters**: ~30 million  
- **Embedding Dimension**: 300  
- **Layers**: 2 Encoder + 2 Decoder  
- **Vocabulary Size**: 25,000  
- **Maximum Sequence Length**: 40  

---

## Project Structure

```bash
English-Hindi-Translation/
│
├─ app.py                
├─ main.py
├─ en_hi_weights.h5
├─ requirements.txt
├─ .gitattributes
├─ .gitignore
├─ .python-version
├─ render.yaml
├─ source_vocab.json
├─ target_vocab.json
├─ transformer.py
├─ utils.py
│
└─ frontend/
    ├─ index.html
    ├─ script.js
    └─ style.css

```
## WorkFlow
```
                                Input Sentence (English)
                                          │
                                          ▼
                    Text Preprocessing (lowercase, clean, tokenize)
                                          │
                                          ▼
                        Vectorization (word → index mapping)
                                          │
                                          ▼
                             Embedding + Positional Encoding
                                          │
                                          ▼
                                       ENCODER
                                  ┌──────────────┐
                                  │ Encoder L1   │
                                  │ Self-Attn +  │
                                  │ FeedForward  │
                                  └──────────────┘
                                          │
                                  ┌──────────────┐
                                  │ Encoder L2   │
                                  │ Self-Attn +  │
                                  │ FeedForward  │
                                  └──────────────┘
                                          │
                                          ▼
                        Contextual Representation(Attention Scores)
                                          │
                                          ▼
                            DECODER (Input: [starttoken])
                                          │
                                          ▼
                                  ┌──────────────┐
                                  │ Decoder L1   │
                                  │ Masked +     │
                                  │ Cross Attn + │
                                  │ FeedForward  │
                                  └──────────────┘
                                          │
                                  ┌──────────────┐
                                  │ Decoder L2   │
                                  │ Masked +     │
                                  │ Cross Attn + │
                                  │ FeedForward  │
                                  └──────────────┘
                                          │
                                          ▼
                                    Dense + Softmax
                                          │
                                          ▼
                                Next Token Prediction
                                          │
                                          ▼
                           Greedy Decoding Loop (until [endtoken])
                                          │
                                          ▼
                                Final Hindi Translation
```
## Installation 
1. Clone the Repository
```
git clone https://github.com/your-username/English-Hindi-Translation.git
cd English-Hindi-Translation
```
3. Install Dependencies
```
pip install -r requirements.txt
```
5. Run the Streamlit App
```
streamlit run app.py
```
