import tensorflow as tf

from tensorflow.keras.layers import Layer, Dense, LayerNormalization
from tensorflow.keras import Model

from numpy import pi

def get_mask(size):
    """Create a mask that prevents previous locations from attending to future locations"""
    
    i = tf.range(size)[:, None]
    j = tf.range(size)
    
    # For each element of i, check if it is greater or equal to each element of j to create the mask
    mask = i >= j
    mask = tf.cast(mask, tf.float32)
    
    return mask

def gelu(x):
    """Smooth ReLU activation"""
    
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / pi) * (x + 0.044715 * tf.pow(x, 3))))

def expand_tile(x, axis, size):
    """Create a new dimension and scale it"""
    
    x = tf.expand_dims(x, axis)
    x = tf.tile(x, [size if axis_ == axis else 1 for axis_ in range(tf.rank(x))])
    
    return x

class MultiHeadAttention(Layer):
    def __init__(self, num_heads):
        super(MultiHeadAttention, self).__init__()
        
        # The number of heads
        self.num_heads = num_heads
    
    def build(self, input_shape):
        # The parameters are initialized here instead of __init__ because they depend on the input_shape
        
        """Initialize remaining parameters"""
        
        # fully-connected neural network layers that expands the input and compress the output
        self.expander = Dense(input_shape[-1] * 3, activation="linear")
        self.compressor = Dense(input_shape[-1], activation="linear")
    
    def attention(self, queries, keys, values):
        # Queries, keys and values all have shape: (batch_size, heads, seq_length, head_features)
        
        """Calculate attention weights and mask them"""
        
        # Compute the weights by using the dot-product score function
        weights = tf.matmul(queries, keys, transpose_b=True)
        # Scale the weights by the square root of the features
        weights = weights * tf.math.rsqrt(tf.cast(values.shape[-1], tf.float32))
    
        # Mask the attention weights
        mask = tf.reshape(get_mask(values.shape[-2]), [1, 1, values.shape[-2], values.shape[-2]])
        weights = weights * mask - 1e10 * (1 - mask)
        
        # Convert the weights to a probability distribution
        weights = tf.nn.softmax(weights)
        
        # Calculate the context as a weighted sum of the values
        context = tf.matmul(weights, values)
        
        return context
    
    def split_heads(self, v):
        # V has shape: (batch_size, seq_length, features)
        
        return tf.transpose(tf.reshape(v, [v.shape[0], v.shape[1], self.num_heads, int(v.shape[-1]/self.num_heads)]), [0, 2, 1, 3])
    
    def split_input(self, x):
        # X has shape: (batch_size, seq_length, features)
        
        """Apply transformations on x and split it into the queries, keys and values for each head"""
        
        # Features of input before transformation
        features = x.shape[-1]
        
        # Expand the input
        x = self.expander(x)
        
        # Split the input into the queries, keys and values of every head
        queries, keys, values = map(self.split_heads, tf.split(x, 3, axis=-1))
        
        return queries, keys, values
    
    def merge_context(self, context):
        # Context has shape: (1, batch_size, heads, seq_length, head_features)
        
        """Merge each head's result and apply a transformation"""
        
        # Merge all heads
        context = tf.transpose(context, [0, 2, 1, 3])
        context = tf.reshape(context, [context.shape[0], context.shape[1], -1])
        
        # Apply transformation
        context = self.compressor(context)
        
        return context
    
    def call(self, x):
        # X has shape: (batch_size, seq_length, features)
        
        """Perform the full multi-head attention process"""
    
        # Split the input
        queries, keys, values = self.split_input(x)
        
        # Compute the context
        context = self.attention(queries, keys, values)
        
        # Compress into the final output
        output = self.merge_context(context)
        
        return output
    
class PositionWise(Layer):
    def build(self, input_shape):
        # The parameters are initialized here instead of __init__ because they depend on the input_shape
        
        """Initialize the network's parameters"""
        
        self.dense1 = Dense(input_shape[-1] * 4, activation="linear")
        self.dense2 = Dense(input_shape[-1], activation="linear")
    
    def call(self, x):
        # X has shape: (batch_size, seq_length, features)
        
        """Compute the layer's output"""
        
        # Compute first layer's activation with GeLU activation function
        h = self.dense1(x)
        h = gelu(h)
        
        # Compute second's layer activation with linear activation function
        h = self.dense2(h)
        
        return h

class TransformerBlock(Layer):
    def __init__(self, num_heads):
        super(TransformerBlock, self).__init__()
        
        # The multi-head attention layer and norm
        self.attn = MultiHeadAttention(num_heads)
        self.attn_norm = LayerNormalization(epsilon=1e-5)
        
        # The position-wise feedforward layer and norm
        self.position_wise = PositionWise() 
        self.position_wise_norm = LayerNormalization(epsilon=1e-5)
        
    def call(self, x):
        # X has shape: (batch_size, seq_length, features)
        
        """Compute the Transformer block's output"""
        
        # Apply layer normalization and perform multi-headed attention
        attn_output = self.attn(self.attn_norm(x))
        
        # Skip connection
        x = x + attn_output

        # Apply layer normalization and compute position-wise output
        position_wise_output = self.position_wise(self.position_wise_norm(x))
        
        # Skip connection
        x = x + position_wise_output
        
        return x
    
class Embedding(Layer):
    def __init__(self, vocab_size, max_words, embedding_size):
        super(Embedding, self).__init__()
        
        # Embedding matrix describing an embedding for every word index and every position
        self.token_embedding = self.add_weight("token_embedding", shape=[vocab_size, embedding_size], dtype=tf.float32)
        self.position_embedding = self.add_weight("pos_embedding", shape=[max_words, embedding_size], dtype=tf.float32)
    
    def call(self, indices):
        # Indices is a tensor of integers of shape: (batch_size, seq_length)
        
        """Convert a tensor of word indices into the actual embeddings"""
        
        # Compute tokens based on indices
        tokens = tf.gather(self.token_embedding, indices)
        
        # Compute position embeddings based on seq_length
        positions = tf.gather(self.position_embedding, tf.range(indices.shape[1]))
        # Duplicate positions batch_size times
        positions = expand_tile(positions, axis=0, size=indices.shape[0])
        
        return tokens + positions

class GPT2(Model):
    def __init__(self, num_blocks, num_heads, vocab_size, max_words, embedding_size):
        super(GPT2, self).__init__()
        
        # The word embedding converts indices to word vectors
        self.word_embedder = Embedding(vocab_size, max_words, embedding_size)
        
        # Initialize all transformer blocks
        self.blocks = []
        
        for i in range(num_blocks):
            self.blocks.append(TransformerBlock(num_heads))
        
        # This layer normalizes the last block's output
        self.final_norm = LayerNormalization(epsilon=1e-5)
        
    def call(self, x, transfer_learning=False):
        # X has shape: (batch_size, seq_length)
        
        """output last block activation"""
        
        # Place to store block activations
        block_activations = []
        
        # Compute embeddings based on word indices
        activations = self.word_embedder(x)
        
        # Pass the embeddings through each Transformer block
        for block in self.blocks:
            activations = block(activations)
            
            # Store current block activation
            block_activations.append(activations)
        
        # Perform the final normalization
        activations = self.final_norm(activations)
        
        # Split activations to get last_activation
        *_, last_activation = tf.squeeze(tf.split(activations, x.shape[-1], axis=-2), [-2])
        
        # If transfer learning is disabled, predict the next token
        if (not transfer_learning):
            last_activation = tf.matmul(last_activation, self.word_embedder.token_embedding, transpose_b=True)
        
        return last_activation
