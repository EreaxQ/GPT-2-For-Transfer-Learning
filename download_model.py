import os
import sys
from shutil import rmtree, move

import requests

from tqdm import tqdm

# Disable TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from load_model import v1_to_v2

# Note: this file is mainly copied from OpenAI's GPT-2 repository, a few variables were changed and the model conversion was added

# Check if provided command line arguments are likely to follow the correct syntax
if len(sys.argv) != 2:
    print('You must enter the model name as a parameter, e.g.: download_model.py 124M')
    sys.exit(1)

# Get model size
model_size = sys.argv[1]

# Get the directory in which the model's parameters will be stored
subdir = "models/%s" % model_size

# Make empty directories if they don't exist
if not os.path.exists(subdir):
    os.makedirs(subdir)
if not os.path.exists("temp"):
    os.makedirs("temp")

# Fetch all necessary parameters to build the model
for filename in ['checkpoint','encoder.json','hparams.json', 'model.ckpt.data-00000-of-00001', 'model.ckpt.index', 'model.ckpt.meta', 'vocab.bpe']:
    r = requests.get("https://storage.googleapis.com/gpt-2/" + subdir + "/" + filename, stream=True)
    
    # Determine where the parameters will be stored
    storage_path = "temp/%s" % filename
        
    # Write all the parameters into their files
    with open(storage_path, 'wb') as f:
        file_size = int(r.headers["content-length"])
        chunk_size = 1000
        
        # Display loading bar and write chunks into their files
        with tqdm(ncols=100, desc="Fetching " + filename, total=file_size, unit_scale=True) as pbar:
            # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                pbar.update(chunk_size)

# Convert model files to the right format
print("Converting model files...")
v1_to_v2("temp", subdir)

# Move encoder.json and vocab.bpe as they are already in the right format
move("temp/encoder.json", subdir)
move("temp/vocab.bpe", subdir)

# Delete temporary folder with files of the wrong format
rmtree("temp")

print("Done!")

