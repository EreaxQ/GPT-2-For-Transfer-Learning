import tensorflow as tf
import json
import os

from tensorflow.python.training import py_checkpoint_reader
from model import GPT2

def path_join(*args):
    """Better path joining"""
    
    return os.path.join(*args).replace("\\", "/")

def load_weights(scope, variable_names, layer, reader):
    """Load all necessary variables to get a layer working from GPT2 model file"""
    
    # Get the layer's weights data (this is necessary for squeezing useless dimensions or expanding dims)
    prev_weights = layer.get_weights()
    
    # Load all weights iteratively
    weights = []
        
    for i, variable_name in enumerate(variable_names):
        # Get variable from reader
        weights.append(tf.reshape(reader.get_tensor("%s/%s" % (scope, variable_name)), prev_weights[i].shape))
    
    # Set the layer's weights to the loaded variables
    layer.set_weights(weights)

def load_model(model_path):
    """Load GPT2 model from TF2 save file"""
    
    # Load hyperparameters
    hparams = {}
    
    with open(path_join(model_path, "hparams.json"), "r") as file:
        hparams = json.load(file)
    
    # Initialize the GPT2 model
    gpt2 = GPT2(hparams["n_layer"], hparams["n_head"], hparams["n_vocab"], hparams["n_ctx"], hparams["n_embd"])
    
    # Load weights
    gpt2.load_weights(path_join(model_path, "weights"))
    
    return gpt2

def save_model(model, save_path):
    """Save a GPT2 model to a TF2 save file"""
    
    # Save weights
    model.save_weights(path_join(save_path, "weights"))
    
    # Load hyperparameters
    hparams = {
        "n_vocab": model.word_embedder.token_embedding.shape[0],
        "n_ctx": model.word_embedder.position_embedding.shape[0],
        "n_embd": model.word_embedder.token_embedding.shape[1],
        "n_head": model.blocks[0].attn.num_heads,
        "n_layer": len(model.blocks)
    }
    
    # Write hyperparameters
    with open(path_join(save_path, "hparams.json"), "w") as file:
        file.write(json.dumps(hparams))

def v1_to_v2(model_path, save_path):
    """Load the GPT2 model from TF1 checkpoint file and save it using TF2"""
    
    hparams = {}
    # Load hyperparameters
    with open(path_join(model_path, "hparams.json"), "r") as file:
        hparams = json.load(file)
    
    # Initialize the GPT2 model
    gpt2 = GPT2(hparams["n_layer"], hparams["n_head"], hparams["n_vocab"], hparams["n_ctx"], hparams["n_embd"])
    
    # Build the model using fake input
    fake_input = tf.constant([0], shape=[1, 1], dtype=tf.int32)
    _ = gpt2(fake_input)
    
    # Get the checkpoint containing the variables
    ckpt = tf.train.latest_checkpoint(model_path)
        
    # Get the checkpoint reader
    reader = py_checkpoint_reader.NewCheckpointReader(ckpt)
    
    # Load the variables
    load_weights("model", ["wte", "wpe"], gpt2.word_embedder, reader)
    load_weights("model/ln_f", ["g", "b"], gpt2.final_norm, reader)

    for layer_index in range(hparams["n_layer"]):
        load_weights("model/h%d/attn/c_attn" % layer_index, ["w", "b"], gpt2.blocks[layer_index].attn.expander, reader)
        load_weights("model/h%d/attn/c_proj" % layer_index, ["w", "b"], gpt2.blocks[layer_index].attn.compressor, reader)
        load_weights("model/h%d/ln_1" % layer_index, ["g", "b"], gpt2.blocks[layer_index].attn_norm, reader)

        load_weights("model/h%d/mlp/c_fc" % layer_index, ["w", "b"], gpt2.blocks[layer_index].position_wise.dense1, reader)
        load_weights("model/h%d/mlp/c_proj" % layer_index, ["w", "b"], gpt2.blocks[layer_index].position_wise.dense2, reader)
        load_weights("model/h%d/ln_2" % layer_index, ["g", "b"], gpt2.blocks[layer_index].position_wise_norm, reader)
        
    # Save model v2
    save_model(gpt2, save_path)
