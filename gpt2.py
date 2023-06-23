import numpy as np

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def layer_norm(x, g, b, eps: float = 1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    x = (x - mean) / np.sqrt(variance + eps) # epsilon项用来避免计算中的分母为零错误
    return g * x + b

def linear(x, w, b): # [m, in], [in, out], [out] -> [m, out]
    return x @ w + b

def gpt2(inputs, wte, wpe, blocks, ln_f, n_head):
    # token + positional embeddings
    x = wte[inputs] + wpe[range(len(inputs))] # [n_seq] -> [n_seq, n_embd]

    # foward pass through n_layer transformer blocks
    for block in blocks:
        x = transformer_block(x, **blocks, n_head=n_head)
    
    # projection to vcab
    x = layer_norm(x, **ln_f)
    return x @ wte.T

def mha(x, c_attn, c_proj, n_head): 
    pass
def ffn(x, c_fc, c_proj):
    # project up
    a = gelu(linear(x, **c_fc))

    # project back down
    x = linear(a, **c_proj)

    return x

def transformer_block(x, mlp, attn, ln_1, ln_2, n_head): 
    # multi-head causal self attention
    x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)

    # position-wise feed forward network
    x = x + ffn(layer_norm(x, **ln_2), **mlp)

    return x

def generate(inputs, params, n_head, n_tokens_to_generate):
    from tqdm import tqdm

    for _ in tqdm(range(n_tokens_to_generate), "generating"):  # auto-regressive decode loop
        logits = gpt2(inputs, **params, n_head=n_head)  # model forward pass
        next_id = np.argmax(logits[-1])  # greedy sampling
        inputs.append(int(next_id))  # append prediction to input

    return inputs[len(inputs) - n_tokens_to_generate :]  # only return generated ids


def main(prompt: str, n_tokens_to_generate: int = 40, model_size: str = "124M", models_dir: str = "models"):
    from utils import load_encoder_hparams_and_params

    # load encoder, hparams, and params from the released open-ai gpt-2 files
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)

    # encode the input string using the BPE tokenizer
    input_ids = encoder.encode(prompt)

    # make sure we are not surpassing the max sequence length of our model
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]

    # generate output ids
    output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)

    # decode the ids back into a string
    output_text = encoder.decode(output_ids)

    return output_text


if __name__ == "__main__":
    import fire

    fire.Fire(main)