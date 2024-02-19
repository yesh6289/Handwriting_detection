#importing the required libraries


import os
import sys
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from urllib.parse import unquote, urlparse
from urllib.error import HTTPError
from zipfile import ZipFile
import tarfile
import shutil

CHUNK_SIZE = 40960
DATA_SOURCE_MAPPING = 'handwriting-recognition:https%3A%2F%2Fstorage.googleapis.com%2Fkaggle-data-sets%2F818027%2F1400106%2Fbundle%2Farchive.zip%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com%252F20240214%252Fauto%252Fstorage%252Fgoog4_request%26X-Goog-Date%3D20240214T124740Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D53aa680d1beb1cc70f65276f2f88a917b69d1f95cac61ce851a8d52bbd9e15f4454982b540c24e101138fcf8ef3dbb5dc3deddd142837d48478aeb63698b42ac7bfefc5f0df069ff25d27f2f725b132fb8ce089ed28fae7e081df81dfb2d58118c46c3678fbc019846695ae821340ed24598d574263721b155a0f4ad950d6efca76d746897c7a90a9fbf6d2567984663abecb0243e478d60606eb3183a3131be6e55eab9a97b719eb910569bc139c6d37bbfd2dccf0ff26433a7829dab976faa8aeaf1514bff2ac457289550f7359ce7eee035790a303626a02f66c8e3aedb3e43cec26a9a120f7f3a395200fcedfcbdb57006a37148e192dac0a19efbb11ba2'

KAGGLE_INPUT_PATH='/kaggle/input'
KAGGLE_WORKING_PATH='/kaggle/working'
KAGGLE_SYMLINK='kaggle'

!umount /kaggle/input/ 2> /dev/null
shutil.rmtree('/kaggle/input', ignore_errors=True)
os.makedirs(KAGGLE_INPUT_PATH, 0o777, exist_ok=True)
os.makedirs(KAGGLE_WORKING_PATH, 0o777, exist_ok=True)

try:
  os.symlink(KAGGLE_INPUT_PATH, os.path.join("..", 'input'), target_is_directory=True)
except FileExistsError:
  pass
try:
  os.symlink(KAGGLE_WORKING_PATH, os.path.join("..", 'working'), target_is_directory=True)
except FileExistsError:
  pass

for data_source_mapping in DATA_SOURCE_MAPPING.split(','):
    directory, download_url_encoded = data_source_mapping.split(':')
    download_url = unquote(download_url_encoded)
    filename = urlparse(download_url).path
    destination_path = os.path.join(KAGGLE_INPUT_PATH, directory)
    try:
        with urlopen(download_url) as fileres, NamedTemporaryFile() as tfile:
            total_length = fileres.headers['content-length']
            print(f'Downloading {directory}, {total_length} bytes compressed')
            dl = 0
            data = fileres.read(CHUNK_SIZE)
            while len(data) > 0:
                dl += len(data)
                tfile.write(data)
                done = int(50 * dl / int(total_length))
                sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {dl} bytes downloaded")
                sys.stdout.flush()
                data = fileres.read(CHUNK_SIZE)
            if filename.endswith('.zip'):
              with ZipFile(tfile) as zfile:
                zfile.extractall(destination_path)
            else:
              with tarfile.open(tfile.name) as tarfile:
                tarfile.extractall(destination_path)
            print(f'\nDownloaded and uncompressed: {directory}')
    except HTTPError as e:
        print(f'Failed to load (likely expired) {download_url} to path {destination_path}')
        continue
    except OSError as e:
        print(f'Failed to load {download_url} to path {destination_path}')
        continue

print('Data source import complete.')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

"""# Import Libraries"""

import requests
from PIL import Image
import os
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import matplotlib.pyplot as plt
import numpy as np

"""# Load Data"""

MODEL_NAME = "microsoft/trocr-base-handwritten"
processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

test_image = '/kaggle/input/handwriting-recognition/train_v2/train/TRAIN_00014.jpg'
image = Image.open(test_image).convert('RGB')
image

"""# Handwritten Detection"""

inputs = processor(image, return_tensors="pt").pixel_values
generated_ids = model.generate(inputs)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text)

root_dir = '/kaggle/input/handwriting-recognition/test_v2/test/'
image_paths = os.listdir(root_dir)[:250]
len(image_paths)

images = np.array(image_paths)

def visualize_df(df: np.ndarray):
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    np.random.shuffle(df)
    for i, ax in enumerate(axes.ravel()):
        if i < len(df):
            img_path = df[i]
            image = Image.open(root_dir + img_path).convert('RGB')
            inputs = processor(image, return_tensors="pt").pixel_values
            generated_ids = model.generate(inputs)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            ax.imshow(image)
            ax.set_title(generated_text)
            ax.axis('off')

        else:
            ax.axis('off')

    plt.tight_layout()
    plt.show()

visualize_df(images)
