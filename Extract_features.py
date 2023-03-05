from torchvision.models import vgg16, VGG16_Weights
from models import FeatureExtractor
from preprocessing import read_and_preprocess_image,read_caption_file, create_image_captions_map, clean_caption
from preprocessing import tokenizer, extract_images_and_caption, create_word_ix_map_and_vocabulary
import pickle
import torch.nn as nn
import torch

image_tensor = read_and_preprocess_image(".\\", "dog.jpg")

vgg = vgg16(VGG16_Weights.DEFAULT)
model = FeatureExtractor(vgg)


# Read file
PATH = ".\\captions.txt"
BASE_DIR = ".\\Images\\"
caption_text = read_caption_file(path=PATH)

# Extract images and captions and create an image-caption map
images_path, captions = extract_images_and_caption(caption_text)
image_caption_map = create_image_captions_map(images_path, captions)

# Preprocessing of captions
for key in image_caption_map:
    image_caption_map[key] = [clean_caption(caption) for caption in image_caption_map[key]]
    image_caption_map[key] = [tokenizer(caption) for caption in image_caption_map[key]]

# create word-index map and vocabulary and a caption list
captions_list = []
words = []
for key in image_caption_map:
    captions_list.append(image_caption_map[key])
    for sentence in image_caption_map[key]:
        words += sentence

word_ix_map, vocabulary = create_word_ix_map_and_vocabulary(words)

feature_map = {}

# Read and preprocess images
image_tensor_list = []
for image_file in image_caption_map:
    image = read_and_preprocess_image(BASE_DIR, image_file)
    feature_map[image_file] = model(image)

print(f"Extracted features: {len(feature_map)}")

pickle.dump(feature_map, open("features.pkl", "wb"))

