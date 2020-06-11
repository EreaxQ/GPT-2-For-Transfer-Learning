# GPT-2-For-Transfer-Learning
An implementation of GPT-2 that uses TensorFlow 2 to make GPT-2 more accessible.

# Getting started
First, you will need to clone the repository.
```cmd
git clone https://github.com/EreaxQ/GPT-2-For-Transfer-Learning.git gpt2
```
Download TensorFlow:
```cmd
pip install tensorflow==2.1.0
```
Or for faster computation with GPU:
```cmd
pip install tensorflow-gpu==2.1.0
```
Install requirements:
```cmd
cd gpt2
pip install -r requirements.txt
```
### Downloading a model
There are different model sizes to choose from (124M, 355M, 774M, 1558M).
To download a model, use this command:
```cmd
python download_model.py <model_size>
```
For example, to download the original GPT-2 model:
```cmd
python download_model.py 1558M
```
This command will store the model and encoder parameters in a directory called 'models'

### Using the model
To use the model, you just need to load it from the downloaded model files. The model takes as input an integer tensor containing word indices. To convert words to word indices, you will need the encoder.
```python
from gpt2.model import GPT2
from gpt2.encoder import get_encoder

model = load_model("models/1558M")
encoder = get_encoder("models/1558M")

words = "This sentence contains"
word_indices = encoder.encode(words)

next_word_embedding = model(tf.constant(word_indices, shape=[1, len(word_indices)], dtype=tf.int32))
next_word_index = tf.math.argmax(next_word_embedding)

next_word = encoder.decode([next_word_index.numpy()])

print("Context: [%s], Prediction: [%s]" % (words, next_word))
