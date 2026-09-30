"""
Attention Is All You Need: Build the Transformer From Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_token_to_id_vocab
def build_token_to_id_vocab(sentences, specials=('<pad>', '<bos>', '<eos>', '<unk>')):
    
    token_to_id = {}

    for token in specials:
        if token not in token_to_id:
            token_to_id[token] = len(token_to_id)
    
    for sentence in sentences:
        for token in sentence.split():
            if token not in token_to_id:
                token_to_id[token] = len(token_to_id)

    return token_to_id

# Step 2 - build_id_to_token_vocab
def build_id_to_token_vocab(token_to_id):
    # TODO: build the inverse id-to-token dictionary from token_to_id
    id_to_token = {}

    for token, idx in token_to_id.items():
        id_to_token[idx] = token

    return id_to_token

# Step 3 - encode_sentence_to_ids
def encode_sentence_to_ids(sentence, token_to_id, unk_token='<unk>'):
    # TODO: convert whitespace tokens of `sentence` to ids via `token_to_id`, using `unk_token`'s id for OOV

    ids = []

    for token in sentence.split():
        if token in token_to_id:
            ids.append(token_to_id[token])
        
        else:
            ids.append(token_to_id[unk_token])
        
    return ids

# Step 4 - decode_ids_to_tokens
def decode_ids_to_tokens(ids, id_to_token):
    # TODO: map each id in ids to its token string via id_to_token and return the list
    
    tokens = []


    for idx in ids:
        tokens.append(id_to_token[idx])

    return tokens

# Step 5 - pad_id_sequence
def pad_id_sequence(ids, max_len, pad_id):
    # TODO: return a list of length exactly max_len, padding with pad_id or truncating.
    
    if len(ids) > max_len:
        return ids[:max_len]
        
    else:
        num_padding = max_len - len(ids)
            
        return ids + [pad_id] * num_padding

# Step 6 - stack_padded_sequences_to_batch
import torch

def stack_padded_sequences_to_batch(padded_sequences):
    """Stack a list of equal-length padded id sequences into a 2D LongTensor batch."""
    # TODO: stack padded id sequences into a (B, L) torch.long tensor

    batch_t = torch.tensor(padded_sequences, dtype=torch.long)

    return batch_t

# Step 7 - scale_embeddings_by_sqrt_d_model
import math
import torch

def scale_embeddings_by_sqrt_d_model(embeddings, d_model):
    """Scale a token embedding tensor by sqrt(d_model)."""
    # TODO: rescale embeddings by sqrt(d_model) as in the original Transformer paper


    scale = d_model ** .5

    rescale = torch.mul(embeddings, scale)

    return rescale

# Step 8 - compute_positional_div_term
import torch

def compute_positional_div_term(d_model):
    # TODO: return a 1D FloatTensor of length d_model // 2 holding the sinusoidal frequency divisors
    
    even_indices = torch.arange(0, d_model, 2, dtype=torch.float32)
    
    div_v = torch.exp(
        -even_indices * (math.log(10000.0) / d_model))

    return div_v

# Step 9 - build_position_index_column
import torch

def build_position_index_column(max_len):
    """Return a (max_len, 1) float tensor of [0, 1, ..., max_len-1]."""
    # TODO: build a column vector of position indices from 0 to max_len-1

    pos_vec = torch.arange(max_len, dtype=torch.float32)

    col_pos = pos_vec[:, None]

    return col_pos

# Step 10 - fill_even_indices_with_sin
import torch

def fill_even_indices_with_sin(pe, position, div_term):
    """Fill even feature indices of pe with sin(position * div_term)."""
    # TODO: write sin(position * div_term) into the even-indexed columns of pe and return it
    pass

    pe[:,0::2] = torch.sin(position * div_term)

    return pe

# Step 11 - fill_odd_indices_with_cos
import torch

def fill_odd_indices_with_cos(pe, position, div_term):
    # TODO: fill the odd-indexed columns of pe with cos(position * div_term)
    

    pe[:,1::2] = torch.cos(position * div_term)

    return pe

# Step 12 - build_sinusoidal_positional_encoding
import torch

def build_sinusoidal_positional_encoding(max_len, d_model):
    """Assemble the (max_len, d_model) sinusoidal positional encoding matrix."""
    # TODO: build the (max_len, d_model) sinusoidal positional encoding matrix
    
    pe = torch.zeros(max_len, d_model, dtype=torch.float32)

    position = torch.arange(
        0, max_len, dtype=torch.float32
    ).unsqueeze(1)

    div_term = compute_positional_div_term(d_model)

    pe = fill_even_indices_with_sin(pe, position, div_term)
    pe = fill_odd_indices_with_cos(pe, position, div_term)

    return pe

# Step 13 - add_positional_encoding_to_embeddings
import torch

def add_positional_encoding_to_embeddings(embedded_batch, positional_encoding):
    # TODO: add the first L rows of positional_encoding to embedded_batch and return the sum.
    

    L = embedded_batch.shape[1]

    return embedded_batch + positional_encoding[:L]

# Step 14 - build_padding_mask
import torch

def build_padding_mask(token_ids, pad_id):
    """Return a (B, 1, 1, L) bool mask: True where token_ids != pad_id."""
    # TODO: build a boolean mask marking non-pad positions, shaped for broadcasting against attention score
    
    mask = token_ids != pad_id

    mask = mask.unsqueeze(1).unsqueeze(2)

    return mask

# Step 15 - build_causal_mask
import torch

def build_causal_mask(seq_len):
    """Return a (1, 1, seq_len, seq_len) bool mask, True on and below diagonal."""
    # TODO: build a lower-triangular boolean causal mask of shape (1, 1, seq_len, seq_len)
    
    mask_lower_tri = torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool))

    mask_lower_tri = mask_lower_tri.unsqueeze(0).unsqueeze(0)

    return mask_lower_tri

# Step 16 - combine_padding_and_causal_masks
import torch

def combine_padding_and_causal_masks(padding_mask, causal_mask):
    # TODO: combine a (B,1,1,L) padding mask with a (1,1,L,L) causal mask into (B,1,L,L).
    
    combined = torch.logical_and(padding_mask, causal_mask)

    return combined

# Step 17 - compute_raw_attention_scores
import torch

def compute_raw_attention_scores(query, key):
    """Compute raw attention scores Q @ K^T over the last two dimensions."""
    # TODO: matmul query with the transpose of key over the last two axes
    
    k_tran = key.transpose(-2, -1)

    attention = query @ k_tran

    return attention

# Step 18 - scale_attention_scores
import torch
import math

def scale_attention_scores(scores, d_k):
    # TODO: divide raw attention scores by sqrt(d_k) to stabilize softmax inputs
    
    scal_fac = d_k ** .5

    attention_scores = scores / scal_fac

    return attention_scores

# Step 19 - mask_attention_scores_with_neg_inf
import torch

def mask_attention_scores_with_neg_inf(scores, mask):
    """Set entries of scores where mask is False to -inf."""
    # TODO: replace blocked positions of scores with negative infinity
    
    return scores.masked_fill(~mask, float("-inf"))

# Step 20 - softmax_attention_weights
import torch

def softmax_attention_weights(masked_scores):
    # TODO: softmax over the last axis, zeroing rows that are entirely -inf
    

    softmax = F.softmax(masked_scores, dim=-1)

    softmax = torch.nan_to_num(softmax, nan=0.0)

    return softmax

# Step 21 - apply_attention_weights_to_values
import torch

def apply_attention_weights_to_values(attention_weights, value):
    """Multiply attention weights by the value matrix to produce context vectors."""
    # TODO: combine attention weights (..., Lq, Lk) with value (..., Lk, d_v)
    
    att_w_v = attention_weights @ value

    return att_w_v

# Step 22 - scaled_dot_product_attention
import torch

def scaled_dot_product_attention(query, key, value, mask=None):
    """Run scaled dot-product attention; return (context, attention_weights)."""
    # TODO: chain raw scores, scale by sqrt(d_k), optionally mask, softmax, then mix values
    Q = query
    V = value
    K = key
    
    
    d_k = Q.shape[-1]

    scores = Q @ K.transpose(-2, -1)
    
    scores = scores / (d_k ** .5)

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    attention = torch.nn.functional.softmax(scores, dim=-1)

    attention = torch.nan_to_num(
        attention, nan=0.0
    )

    context = attention @ value

    return context, attention

# Step 23 - split_last_dim_into_heads
import torch

def split_last_dim_into_heads(tensor, num_heads):
    # TODO: reshape (B, L, d_model) into (B, L, num_heads, d_model // num_heads)
    B, L, d_model = tensor.shape
        
    assert d_model % num_heads == 0

    d_head = d_model // num_heads

    tensor = tensor.reshape(B, L, num_heads, d_head)

    return tensor

# Step 24 - transpose_heads_before_sequence
import torch

def transpose_heads_before_sequence(split_tensor):
    # TODO: rearrange (B, L, num_heads, d_k) into (B, num_heads, L, d_k).
    
    split_tensor = split_tensor.transpose(1, 2)

    return split_tensor

# Step 25 - merge_heads_back_to_model_dim
import torch

def merge_heads_back_to_model_dim(multi_head_tensor):
    # TODO: merge the head axis back into the feature axis to reconstruct d_model
    
    B, num_heads, L, d_k = multi_head_tensor.shape
    
    d_model = num_heads * d_k
    
    multi_head_tensor = multi_head_tensor.transpose(1, 2)

    multi_head_tensor = multi_head_tensor.contiguous()
    
    multi_head_tensor = multi_head_tensor.reshape(B, L, d_model)

    return multi_head_tensor

# Step 26 - apply_linear_projection
def apply_linear_projection(x, weight, bias):
    # TODO: return x @ weight^T + bias (bias may be None) with shape (..., out_features)

    weight = weight.transpose(0, -1)

    lin_proj = x @ weight

    if bias is not None:
        output = lin_proj + bias
    else:
        output = lin_proj
    
    return output

# Step 27 - project_to_query_key_value
def project_to_query_key_value(x, w_q, b_q, w_k, b_k, w_v, b_v):
    # TODO: project x into separate query, key, and value tensors via three linear layers
    

    Q = apply_linear_projection(x, w_q, b_q)

    K = apply_linear_projection(x, w_k, b_k)

    V = apply_linear_projection(x, w_v, b_v)

    return Q, K, V

# Step 28 - split_qkv_into_heads
import torch

def split_qkv_into_heads(q, k, v, num_heads):
    # TODO: split each of q, k, v into (B, num_heads, L, d_k) and return as a tuple

   q = split_last_dim_into_heads(q, num_heads)
   k = split_last_dim_into_heads(k, num_heads)
   v = split_last_dim_into_heads(v, num_heads)

   q = transpose_heads_before_sequence(q)
   k = transpose_heads_before_sequence(k)
   v = transpose_heads_before_sequence(v)

   return q, k, v

# Step 29 - multi_head_scaled_dot_product_attention
import torch

def multi_head_scaled_dot_product_attention(q_h, k_h, v_h, mask=None):
    # TODO: run scaled dot-product attention over per-head Q, K, V and return (context, weights)
    Q = q_h
    K = k_h
    V = v_h
    
    d_k = Q.shape[-1]

    scores = Q @ K.transpose(-2, -1)
    
    scores = scores / (d_k ** .5)

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    attention = torch.nn.functional.softmax(scores, dim=-1)

    attention = torch.nan_to_num(
        attention, nan=0.0
    )

    context = attention @ V

    return context, attention

# Step 30 - merge_heads_and_project_output
import torch

def merge_heads_and_project_output(context, w_o, b_o):
    # TODO: merge the head axis back into d_model and apply the output linear projection.
    
   context = merge_heads_back_to_model_dim(context)

   lin_proj = apply_linear_projection(context, w_o, b_o)

   return lin_proj

# Step 31 - assemble_multi_head_attention_forward
def assemble_multi_head_attention_forward(query, key, value, w_q, w_k, w_v, w_o, num_heads, mask=None):
    # TODO: project Q/K/V, split into heads, run scaled dot-product attention, merge heads, output projection.

    Q = torch.nn.functional.linear(query, w_q)
    K = torch.nn.functional.linear(key, w_k)
    V = torch.nn.functional.linear(value, w_v)


    B, lq, d_model = Q.shape
    lk = K.shape[1]

    assert d_model % num_heads == 0

    d_k = d_model // num_heads

    Q = Q.reshape(B, lq, num_heads, d_k).transpose(1, 2)
    K = K.reshape(B, lk, num_heads, d_k).transpose(1, 2)
    V = V.reshape(B, lk, num_heads, d_k).transpose(1, 2)


    scores = Q @ K.transpose(-1, -2)

    scores = scores / (d_k ** .5)

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    attention_weights = torch.softmax(scores, dim=-1)

    attention_weights = torch.nan_to_num(attention_weights, nan=0.0)

    context = attention_weights @ V 

    context = context.transpose(1, 2).contiguous()

    context = context.reshape(B, lq, d_model)

    output = torch.nn.functional.linear(context, w_o)

    return output

# Step 32 - apply_ffn_first_linear_and_relu
def apply_ffn_first_linear_and_relu(x, w1, b1):
    # TODO: project x by w1, add b1, then apply a ReLU activation.   
    l1 = (x @ w1) + b1

    actv = torch.relu(l1)

    return actv

# Step 33 - apply_ffn_second_linear
import torch

def apply_ffn_second_linear(hidden, w2, b2):
    # TODO: project hidden (..., d_ff) back to (..., d_model) via w2 and b2.
    
    l2 = (hidden @ w2) + b2

    return l2

# Step 34 - position_wise_feed_forward_network
def position_wise_feed_forward_network(x, w1, b1, w2, b2):
    # TODO: compose the two FFN linears with a ReLU in between, returning shape (B, T, d_model).
    pos_trans_l1 = apply_ffn_first_linear_and_relu(x, w1, b1)

    pos_trans_l2 = apply_ffn_second_linear(pos_trans_l1, w2, b2)

    return pos_trans_l2

# Step 35 - compute_layer_norm_mean_and_variance
import torch

def compute_layer_norm_mean_and_variance(x):
    # TODO: return (mean, variance) reduced over the last dim with shape (..., 1)
    mean = torch.mean(x, keepdim=True, dim=-1)

    variance = torch.var(x, keepdim=True, dim=-1, correction=0)


    return mean, variance

# Step 36 - normalize_and_scale_with_gamma_beta
import torch

def normalize_and_scale_with_gamma_beta(x, gamma, beta, eps=1e-5):
    # TODO: standardize x along the last axis then apply gamma and beta affine transform
    mean, variance = compute_layer_norm_mean_and_variance(x)


    x_hat = (x - mean) / torch.sqrt(variance + eps)

    y = (gamma * x_hat) + beta

    return y

# Step 37 - apply_residual_add_and_norm
import torch

def apply_residual_add_and_norm(residual_input, sublayer_output, gamma, beta, eps=1e-5):
    # TODO: combine the residual with the sublayer output and layer-normalize the result.
    
    wrapped_s = residual_input + sublayer_output

    res = normalize_and_scale_with_gamma_beta(wrapped_s, gamma, beta, eps=1e-5)

    return res

# Step 38 - apply_dropout_with_keep_mask
def apply_dropout_with_keep_mask(x, keep_mask, keep_prob):
    # TODO: multiply x by the boolean keep_mask and rescale by 1/keep_prob.
    
    scale = 1 / keep_prob
    
    return x * keep_mask.to(x.dtype) * scale

# Step 39 - encoder_layer_self_attention_sublayer
def encoder_layer_self_attention_sublayer(x, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):
    # TODO: run multi-head self-attention on x and wrap with residual add-and-norm.
    
    attention_output = assemble_multi_head_attention_forward(x, x, x, w_q, w_k, w_v, w_o, num_heads, src_mask)
    
    # 2. Wrap with residual add-and-norm (passing the original input x for the residual connection)
    
    return apply_residual_add_and_norm(x, attention_output, gamma, beta, eps=1e-5)

# Step 40 - encoder_layer_feed_forward_sublayer
def encoder_layer_feed_forward_sublayer(x, w1, b1, w2, b2, gamma, beta):
    # TODO: run the position-wise FFN on x and wrap it with residual add-and-norm.
    
    e_FFN = position_wise_feed_forward_network(x, w1, b1, w2, b2)

    return apply_residual_add_and_norm(x, e_FFN, gamma, beta, eps=1e-5)

# Step 41 - assemble_encoder_layer
def assemble_encoder_layer(x, layer_params, num_heads, src_mask):
    # TODO: chain the self-attention sublayer and the feed-forward sublayer using layer_params.
    vals_sh = [layer_params[k] for k in ["w_q", "w_k", "w_v", "w_o", "attn_gamma", "attn_beta"]]
    vals_ffn = [layer_params[k] for k in ["w1", "b1", "w2", "b2", "ffn_gamma", "ffn_beta"]]

    sa_s = encoder_layer_self_attention_sublayer(x, *vals_sh, num_heads, src_mask)
    return encoder_layer_feed_forward_sublayer(sa_s, *vals_ffn)

# Step 42 - stack_encoder_layers
def stack_encoder_layers(x, encoder_layer_params_list, num_heads, src_mask):
    # TODO: sequentially apply each encoder layer to the running hidden state and return the final tensor.
    
    hidden_state = x
   
    
    
    for param in encoder_layer_params_list:
        hidden_state = assemble_encoder_layer(hidden_state, param, 
        num_heads=num_heads, src_mask=src_mask)

    return hidden_state

# Step 43 - decoder_layer_masked_self_attention_sublayer
import torch

def decoder_layer_masked_self_attention_sublayer(y, w_q, w_k, w_v, w_o, gamma, beta, num_heads, tgt_mask):
    # TODO: run masked multi-head self-attention on y and wrap with residual add-and-norm.
    
    attention_score = assemble_multi_head_attention_forward(y, y, y, w_q, w_k, w_v, 
    w_o, num_heads, tgt_mask)

    return apply_residual_add_and_norm(y, attention_score, gamma, beta, eps=1e-5)

# Step 44 - decoder_layer_cross_attention_sublayer
import torch

def decoder_layer_cross_attention_sublayer(y, encoder_output, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):
    # TODO: run multi-head cross-attention (Q from y, K/V from encoder_output) and wrap with add-and-norm
    if src_mask is not None and src_mask.dim() == 2:
        src_mask = src_mask[:, None, None, :]

    attention_score = assemble_multi_head_attention_forward(y, encoder_output, encoder_output, w_q, w_k, w_v, w_o, num_heads, src_mask)

    return apply_residual_add_and_norm(y, attention_score, gamma, beta, eps=1e-5)

# Step 45 - decoder_layer_feed_forward_sublayer
import torch

def decoder_layer_feed_forward_sublayer(y, w1, b1, w2, b2, gamma, beta):
    # TODO: run the position-wise FFN on y and wrap it with residual add-and-norm
    d_FFN = position_wise_feed_forward_network(y, w1, b1, w2, b2)

    return apply_residual_add_and_norm(y, d_FFN, gamma, beta, eps=1e-5)

# Step 46 - assemble_decoder_layer
def assemble_decoder_layer(y, encoder_output, layer_params, num_heads, src_mask, tgt_mask):
    """Run a full decoder layer: masked self-attention, cross-attention, then FFN.

    layer_params keys (all torch tensors):
      masked self-attention : w_q_self, w_k_self, w_v_self, w_o_self, self_gamma, self_beta
      cross-attention       : w_q_cross, w_k_cross, w_v_cross, w_o_cross, cross_gamma, cross_beta
      feed-forward          : w1, b1, w2, b2, ffn_gamma, ffn_beta
    """
    # TODO: chain the three decoder sublayers using params from layer_params.
    msa_params = [
        layer_params[k]
        for k in [
            "w_q_self", "w_k_self", "w_v_self",
            "w_o_self", "self_gamma", "self_beta"
        ]
    ]

    ca_params = [
        layer_params[k]
        for k in [
            "w_q_cross", "w_k_cross", "w_v_cross",
            "w_o_cross", "cross_gamma", "cross_beta"
        ]
    ]

    ffn_params = [
        layer_params[k]
        for k in [
            "w1", "b1", "w2", "b2",
            "ffn_gamma", "ffn_beta"
        ]
    ]

    y = decoder_layer_masked_self_attention_sublayer(
        y, *msa_params, num_heads, tgt_mask
    )

    y = decoder_layer_cross_attention_sublayer(
        y, encoder_output, *ca_params, num_heads, src_mask
    )

    y = decoder_layer_feed_forward_sublayer(
        y, *ffn_params
    )

    return y

# Step 47 - stack_decoder_layers
def stack_decoder_layers(y, encoder_output, decoder_layer_params_list, num_heads, src_mask, tgt_mask):
    # TODO: sequentially apply each decoder layer to the running target hidden state.
    

    hidden_state = y

    for params in decoder_layer_params_list:
        hidden_state = assemble_decoder_layer(hidden_state, encoder_output, params,
        num_heads=num_heads, src_mask=src_mask, tgt_mask=tgt_mask)
    
    return hidden_state

# Step 48 - apply_final_output_projection
def apply_final_output_projection(decoder_output, output_projection_weight, output_projection_bias=None):
    # TODO: project decoder hidden states (B, T, D) to vocabulary logits (B, T, V).

    return apply_linear_projection(decoder_output, output_projection_weight, output_projection_bias)

# Step 49 - tie_output_projection_to_token_embeddings
import torch

def tie_output_projection_to_token_embeddings(token_embedding_weight):
    """Return an output projection weight that shares storage with token_embedding_weight.

    Input shape: (vocab_size, d_model). Output shape: (d_model, vocab_size).
    """
    # TODO: return an output projection weight tied to the token embedding matrix
    
    return token_embedding_weight.transpose(-1,0)

# Step 50 - apply_log_softmax_over_vocab
def apply_log_softmax_over_vocab(logits):
    # TODO: Convert decoder logits (B, T, V) into log probabilities over the vocabulary axis.
    
    logsoftmax = torch.nn.LogSoftmax(dim=-1)

    return logsoftmax(logits)

# Step 51 - run_transformer_forward
def run_transformer_forward(src_ids, tgt_ids, model_params, num_heads, pad_id):
    # TODO: embed src+tgt, add PE, build masks, run encoder/decoder, project to log probs.
    
    # Step 1 Masks
    src_mask = build_padding_mask(src_ids, pad_id)
    
    tgt_padding_mask = build_padding_mask(tgt_ids, pad_id)
    tgt_casual_mask = build_causal_mask(tgt_ids.shape[1])
    tgt_mask = combine_padding_and_causal_masks(tgt_padding_mask, tgt_casual_mask)

    # Step 2 Embeddings
    embedding_weight = model_params["token_embedding"]

    src = torch.nn.functional.embedding(src_ids, embedding_weight)
    tgt = torch.nn.functional.embedding(tgt_ids, embedding_weight)

    # Step 3 Scale
    
    d_model = embedding_weight.shape[1]

    src = scale_embeddings_by_sqrt_d_model(src, d_model)
    tgt = scale_embeddings_by_sqrt_d_model(tgt, d_model)

    # Step 4 Positional Encoding
    max_len = max(src_ids.shape[1], tgt_ids.shape[1])
    pe = build_sinusoidal_positional_encoding(max_len, d_model)

    src = add_positional_encoding_to_embeddings(src, pe)
    tgt = add_positional_encoding_to_embeddings(tgt, pe)

    # Step 5 Encoder/Decoder

    encoder_output = stack_encoder_layers(src, model_params["encoder_layers"], num_heads, src_mask)

    decoder_output = stack_decoder_layers(tgt, encoder_output, model_params["decoder_layers"], num_heads, src_mask, tgt_mask)

    # Step 6 Vocab Proj
    logits = apply_final_output_projection(decoder_output, model_params["output_projection"] )

    # Step 7 Softmax Raw Logits to Get Probs

    log_probs = apply_log_softmax_over_vocab(logits)

    return log_probs

# Step 52 - init_encoder_layer_parameters
import torch
import math

def init_encoder_layer_parameters(d_model, num_heads, d_ff):
    """Return a dict of leaf tensors with requires_grad=True for one encoder layer."""
    # TODO: allocate w_q, w_k, w_v, w_o, w1, b1, w2, b2, attn_gamma, attn_beta, ffn_gamma, ffn_beta.
    w_q = torch.randn(d_model, d_model) / (d_model ** .5)
    w_k = torch.randn(d_model, d_model) / (d_model ** .5)
    w_v = torch.randn(d_model, d_model) / (d_model ** .5)
    w_o = torch.randn(d_model, d_model) / (d_model ** .5)

    w1 = torch.randn(d_model, d_ff) / (d_model ** .5)
    b1 = torch.zeros(d_ff)

    w2 = torch.randn(d_ff, d_model) / (d_ff ** .5)
    b2 = torch.zeros(d_model)

    attn_gamma = torch.ones(d_model)
    attn_beta = torch.zeros(d_model)

    ffn_gamma = torch.ones(d_model)
    ffn_beta = torch.zeros(d_model)
    
    params = {
        "w_q": w_q.requires_grad_(),
        "w_k": w_k.requires_grad_(),
        "w_v": w_v.requires_grad_(),
        "w_o": w_o.requires_grad_(),

        "w1": w1.requires_grad_(),
        "b1": b1.requires_grad_(),
        "w2": w2.requires_grad_(),
        "b2": b2.requires_grad_(),

        "attn_gamma": attn_gamma.requires_grad_(),
        "attn_beta": attn_beta.requires_grad_(),
        "ffn_gamma": ffn_gamma.requires_grad_(),
        "ffn_beta": ffn_beta.requires_grad_(),
    }

    return params

# Step 53 - init_decoder_layer_parameters
import torch
import math

def init_decoder_layer_parameters(d_model, num_heads, d_ff):
    assert d_model % num_heads == 0

    params = {
        # Masked self-attention
        "w_q_self": (torch.randn(d_model, d_model) / math.sqrt(d_model)).requires_grad_(),
        "w_k_self": (torch.randn(d_model, d_model) / math.sqrt(d_model)).requires_grad_(),
        "w_v_self": (torch.randn(d_model, d_model) / math.sqrt(d_model)).requires_grad_(),
        "w_o_self": (torch.randn(d_model, d_model) / math.sqrt(d_model)).requires_grad_(),

        # Cross-attention
        "w_q_cross": (torch.randn(d_model, d_model) / math.sqrt(d_model)).requires_grad_(),
        "w_k_cross": (torch.randn(d_model, d_model) / math.sqrt(d_model)).requires_grad_(),
        "w_v_cross": (torch.randn(d_model, d_model) / math.sqrt(d_model)).requires_grad_(),
        "w_o_cross": (torch.randn(d_model, d_model) / math.sqrt(d_model)).requires_grad_(),

        # Feed-forward network
        "w1": (torch.randn(d_model, d_ff) / math.sqrt(d_model)).requires_grad_(),
        "b1": torch.zeros(d_ff, requires_grad=True),

        "w2": (torch.randn(d_ff, d_model) / math.sqrt(d_ff)).requires_grad_(),
        "b2": torch.zeros(d_model, requires_grad=True),

        # LayerNorm after masked self-attention
        "self_gamma": torch.ones(d_model, requires_grad=True),
        "self_beta": torch.zeros(d_model, requires_grad=True),

        # LayerNorm after cross-attention
        "cross_gamma": torch.ones(d_model, requires_grad=True),
        "cross_beta": torch.zeros(d_model, requires_grad=True),

        # LayerNorm after FFN
        "ffn_gamma": torch.ones(d_model, requires_grad=True),
        "ffn_beta": torch.zeros(d_model, requires_grad=True),
    }

    return params

# Step 54 - init_embedding_and_projection_parameters
import torch

def init_embedding_and_projection_parameters(vocab_size, d_model, tie_weights=True):
    """Allocate src/tgt embeddings and output projection (optionally tied)."""
    src_embedding = torch.randn(
        vocab_size, d_model
    ).requires_grad_()

    tgt_embedding = torch.randn(
        vocab_size, d_model
    ).requires_grad_()

    if tie_weights:
        output_projection = tgt_embedding
    else:
        output_projection = torch.randn(
            vocab_size, d_model
        ).requires_grad_()

    return {
        "src_embedding": src_embedding,
        "tgt_embedding": tgt_embedding,
        "output_projection": output_projection,
    }

# Step 55 - collect_model_parameters_into_list
import torch

def collect_model_parameters_into_list(
    encoder_layer_params,
    decoder_layer_params,
    embedding_params
):
    parameters = []
    seen = set()

    # Encoder layers
    for layer_params in encoder_layer_params:
        for tensor in layer_params.values():
            if id(tensor) not in seen:
                parameters.append(tensor)
                seen.add(id(tensor))

    # Decoder layers
    for layer_params in decoder_layer_params:
        for tensor in layer_params.values():
            if id(tensor) not in seen:
                parameters.append(tensor)
                seen.add(id(tensor))

    # Embeddings / output projection
    for tensor in embedding_params.values():
        if id(tensor) not in seen:
            parameters.append(tensor)
            seen.add(id(tensor))

    return parameters

# Step 56 - shift_targets_right_with_start_token
def shift_targets_right_with_start_token(target_ids, start_token_id):
    # TODO: prepend start_token_id and drop the last column so output shape matches target_ids
    
    start = torch.full(
        (target_ids.size(0), 1),
        start_token_id,
        dtype=target_ids.dtype,
        device=target_ids.device

    )

    return torch.cat([start, target_ids[:, :-1]], dim=-1)

# Step 57 - compute_noam_learning_rate
def compute_noam_learning_rate(step, d_model, warmup_steps):
    # TODO: return the Noam warmup learning rate for the given step.
    
    lr_s = ((d_model ** -.5) * min((step ** -.5), step * (warmup_steps ** -1.5)))

    return lr_s

# Step 58 - build_uniform_smoothing_distribution
import torch

def build_uniform_smoothing_distribution(shape, vocab_size, epsilon):
    # TODO: return a float tensor of `shape` filled with epsilon / (vocab_size - 2).
    
    smoothing = epsilon / (vocab_size - 2)

    return torch.full(shape, smoothing, dtype=torch.float32)

# Step 59 - set_confidence_on_gold_tokens
import torch

def set_confidence_on_gold_tokens(smoothed_distribution, gold_token_ids, confidence):
    """Place confidence mass at gold-token positions of a smoothed target distribution."""
    # TODO: write the confidence value at each gold token id along the vocab axis
    output_distrib = smoothed_distribution.clone()
    index = gold_token_ids.unsqueeze(-1)
    output_distrib.scatter_(dim=-1, index=index, value=confidence)
    return output_distrib

# Step 60 - zero_pad_column_and_pad_token_rows
import torch

def zero_pad_column_and_pad_token_rows(smoothed_distribution, gold_token_ids, pad_id):
    # TODO: zero the pad column and the rows where the gold token equals pad_id
    
    smoothed_distribution[..., pad_id] = 0.0

    pad_rows = gold_token_ids == pad_id
    smoothed_distribution[pad_rows] = 0.0


    return smoothed_distribution

# Step 61 - compute_label_smoothed_kl_loss
import torch

def compute_label_smoothed_kl_loss(log_probabilities, smoothed_distribution):
    """Return the summed KL loss over all (batch, time, vocab) entries."""
    # TODO: combine log_probabilities with the smoothed target distribution into a scalar loss
    mask = smoothed_distribution > 0
    loss = (-smoothed_distribution[mask] * log_probabilities[mask]).sum()

    return loss

# Step 62 - average_loss_over_non_pad_tokens
import torch

def average_loss_over_non_pad_tokens(total_loss, gold_token_ids, pad_id):
    # TODO: divide total_loss by the count of non-pad tokens in gold_token_ids
    
    non_pad_count = (gold_token_ids != pad_id).sum()

    if non_pad_count == 0:
        return total_loss 

    else:
        return total_loss / non_pad_count

# Step 63 - compute_token_accuracy_ignoring_pad
import torch

def compute_token_accuracy_ignoring_pad(log_probabilities, gold_token_ids, pad_id):
    # TODO: argmax over vocab, compare to gold, average over non-pad positions only

    predicted_token_ids = torch.argmax(log_probabilities, dim=-1)

    non_pad_mask = (gold_token_ids != pad_id)

    non_pad_count = non_pad_mask.sum()

    if non_pad_count == 0:
        return torch.tensor(0.0, device=log_probabilities.device)

    correct = (predicted_token_ids == gold_token_ids) & non_pad_mask

    return correct.sum().float() / non_pad_count

# Step 64 - initialize_adam_optimizer_state
import torch

def initialize_adam_optimizer_state(parameter_list):
    """Allocate Adam m, v zero buffers and a step counter t=0."""
    # TODO: allocate zero buffers for first and second moments, plus step counter
    m = [torch.zeros_like(param) for param in parameter_list]
    v = [torch.zeros_like(param) for param in parameter_list]
    t = 0

    return {
        'm': m,
        'v': v,
        't': t
    }

# Step 65 - update_adam_first_moment
import torch

def update_adam_first_moment(m_prev, grad, beta1):
    """Return m_t = beta1 * m_prev + (1 - beta1) * grad."""
    # TODO: apply the Adam first-moment EMA update and return the new tensor
    
    return beta1 * m_prev + (1-beta1) * grad

# Step 66 - update_adam_second_moment (not yet solved)
# TODO: implement

# Step 67 - apply_adam_bias_correction (not yet solved)
# TODO: implement

# Step 68 - compute_adam_parameter_update (not yet solved)
# TODO: implement

# Step 69 - apply_adam_step_to_all_parameters (not yet solved)
# TODO: implement

# Step 70 - zero_all_parameter_gradients (not yet solved)
# TODO: implement

# Step 71 - compute_batch_training_loss (not yet solved)
# TODO: implement

# Step 72 - run_training_step_with_backprop (not yet solved)
# TODO: implement

# Step 73 - run_training_loop_for_steps (not yet solved)
# TODO: implement

# Step 74 - pick_next_token_by_argmax (not yet solved)
# TODO: implement

# Step 75 - compute_length_penalty (not yet solved)
# TODO: implement

# Step 76 - compute_candidate_scores (not yet solved)
# TODO: implement

# Step 77 - select_top_k_candidates (not yet solved)
# TODO: implement

# Step 78 - append_tokens_to_beam_sequences (not yet solved)
# TODO: implement

# Step 79 - mark_finished_beams (not yet solved)
# TODO: implement

# Step 80 - select_best_finished_beam (not yet solved)
# TODO: implement

