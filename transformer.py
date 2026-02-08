import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding, MultiHeadAttention, Dense, LayerNormalization
import numpy as np


@tf.keras.utils.register_keras_serializable()
def positional_encoding(model_size,HINDI_SEQUENCE_LENGTH):
  output=[]
  for pos in range(HINDI_SEQUENCE_LENGTH):
    PE=np.zeros((model_size,))
    for i in range(model_size):
      if i%2==0:
        PE[i]=np.sin(pos/(10000**(i/model_size)))
      else :
        PE[i]=np.cos(pos/(10000**((i-1)/model_size)))
    output.append(tf.expand_dims(PE,axis=0))
  out=tf.concat(output,axis=0)
  out=tf.expand_dims(out,axis=0)

  return tf.cast(out,dtype=tf.float32)


@tf.keras.utils.register_keras_serializable()
class Embeddings(tf.keras.layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, name=None):
        super().__init__(name=name)

        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.token_embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
            mask_zero=True,
            name="embedding"   # ✅ ABSOLUTELY REQUIRED
        )

    def call(self, inputs):
        embedded_token = self.token_embedding(inputs)
        embedded_positions = positional_encoding(
            self.embed_dim, self.sequence_length
        )
        return embedded_token + embedded_positions

    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
        })
        return config


@tf.keras.utils.register_keras_serializable()
class TransformerEncoder(Layer):
    def __init__(self, embed_dim, latent_dim, num_heads):
        super(TransformerEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.multi_head_attention = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = tf.keras.Sequential([
            Dense(latent_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm_1 = LayerNormalization()
        self.layernorm_2 = LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, mask=None):
        attention_output = self.multi_head_attention(
            query=inputs, key=inputs, value=inputs, attention_mask=mask
        )
        out_1 = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(out_1)
        out_2 = self.layernorm_2(out_1 + proj_output)
        return out_2




@tf.keras.utils.register_keras_serializable()
class TransformerDecoder(Layer):
    def __init__(self, embed_dim, latent_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads

        self.self_attention = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.cross_attention = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(latent_dim, activation="relu"),
            Dense(embed_dim)
        ])
        self.ln1 = LayerNormalization()
        self.ln2 = LayerNormalization()
        self.ln3 = LayerNormalization()
        self.supports_masking = True

    def call(self, x, enc_outputs, mask=None, enc_mask=None):
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]

        # causal mask (tokens can only attend to <= i)
        causal_mask = tf.linalg.band_part(tf.ones((T, T)), -1, 0)
        causal_mask = tf.cast(causal_mask, tf.bool)
        causal_mask = tf.reshape(causal_mask, (1, T, T))
        causal_mask = tf.tile(causal_mask, [B, 1, 1])

        if mask is not None:
            mask = tf.cast(mask[:, tf.newaxis, :], tf.bool)
            combined_mask = tf.logical_and(causal_mask, mask)
        else:
            combined_mask = causal_mask


        attn_out = self.self_attention(query=x, key=x, value=x, attention_mask=combined_mask)
        x = self.ln1(x + attn_out)


        if enc_mask is not None:
            enc_mask = tf.cast(enc_mask[:, tf.newaxis, :], tf.bool)
        attn_out = self.cross_attention(query=x, key=enc_outputs, value=enc_outputs, attention_mask=enc_mask)
        x = self.ln2(x + attn_out)

        # feed-forward
        ffn_out = self.ffn(x)
        x = self.ln3(x + ffn_out)

        return x
    
