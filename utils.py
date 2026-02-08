import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, Dense, Dropout, Input
from tensorflow.keras.models import Model, load_model
from transformer import Embeddings, TransformerEncoder, TransformerDecoder
import os
import tensorflow as tf
import json
import numpy as np
import re


VOCAB_SIZE = 25000
ENGLISH_SEQUENCE_LENGTH = 40
HINDI_SEQUENCE_LENGTH = 40
EMBEDDING_DIM = 256
NUM_HEADS = 4
LATENT_DIM = 2048
NUM_LAYERS = 2

def build_transformer():
    encoder_inputs = Input(shape=(None,), dtype="int64", name="input_1")
    x = Embeddings(ENGLISH_SEQUENCE_LENGTH, VOCAB_SIZE, EMBEDDING_DIM)(encoder_inputs)
    for _ in range(NUM_LAYERS):
        x = TransformerEncoder(EMBEDDING_DIM, LATENT_DIM, NUM_HEADS)(x)
    encoder_outputs = x
    
    decoder_inputs = Input(shape=(None,), dtype="int64", name="input_2")
    x = Embeddings(HINDI_SEQUENCE_LENGTH, VOCAB_SIZE, EMBEDDING_DIM)(decoder_inputs)
    for _ in range(NUM_LAYERS):
        x = TransformerDecoder(EMBEDDING_DIM, LATENT_DIM, NUM_HEADS)(x, encoder_outputs)
    
    x = Dropout(0.5)(x)
    decoder_outputs = Dense(VOCAB_SIZE, activation="softmax")(x)
    transformer = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return transformer




def _standardize_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text


def _load_vectorizers():
    with open("source_vocab.json", "r") as f:
        src_vocab_list = json.load(f)
    with open("target_vocab.json", "r") as f:
        tgt_vocab_list = json.load(f)

    src_word_to_index = {w: i for i, w in enumerate(src_vocab_list)}
    tgt_word_to_index = {w: i for i, w in enumerate(tgt_vocab_list)}
    index_to_word = {i: w for i, w in enumerate(tgt_vocab_list)}

    class SimpleVectorizer:
        def __init__(self, word2idx, idx2word=None, max_len=None):
            self.word2idx = word2idx
            self.idx2word = idx2word
            self.max_len = max_len

        def __call__(self, texts):
            tokenized = []
            for t in texts:
                t = _standardize_text(t)
                tokenized.append([self.word2idx.get(w, self.word2idx.get("[UNK]", 1)) for w in t.split()])
            return tf.keras.preprocessing.sequence.pad_sequences(tokenized, maxlen=self.max_len, padding='post')

        def get_vocabulary(self):
            return list(self.word2idx.keys())

    source_vectorizer = SimpleVectorizer(src_word_to_index, max_len=ENGLISH_SEQUENCE_LENGTH)
    target_vectorizer = SimpleVectorizer(tgt_word_to_index, index_to_word, max_len=HINDI_SEQUENCE_LENGTH)
    return source_vectorizer, target_vectorizer, index_to_word

def translator(sentence,transformer):
    source_vectorizer, target_vectorizer, index_to_word = _load_vectorizers()
    transformer.load_weights("en_hi_weights.h5")

    sentence = " ".join(sentence.strip().split())
    words = sentence.split()
    truncated_note = ""
    if len(words) > ENGLISH_SEQUENCE_LENGTH:
        sentence = " ".join(words[:ENGLISH_SEQUENCE_LENGTH])
        truncated_note = f"Input truncated to {ENGLISH_SEQUENCE_LENGTH} tokens for translation."

    src_tokens = source_vectorizer([sentence])
    shifted_target = ["starttoken"]
    output = []

    for _ in range(HINDI_SEQUENCE_LENGTH):
        tgt_text = " ".join(shifted_target)
        tgt_tokens = target_vectorizer([tgt_text])
        logits = transformer([src_tokens, tgt_tokens], training=False)
        next_id = tf.argmax(logits[0, len(shifted_target)-1, :]).numpy()
        next_word = index_to_word.get(next_id, "[UNK]")
        if next_word in ["endtoken", "[UNK]"]:
            break
        output.append(next_word)
        shifted_target.append(next_word)

    translation_text = " ".join(output)
    return translation_text, truncated_note