from PIL import Image
from torchvision import transforms
from nltk.tokenize import word_tokenize
import string
import matplotlib.pyplot as plt


def read_caption_file(path):
    with open(path, "r") as F:
        text_lines = F.read().splitlines()
    return text_lines


def extract_images_and_caption(text_lines):
    images = []
    captions = []
    for line in text_lines:
        split_line = line.split(",")
        image = split_line[0]
        caption = "".join(split_line[1:])
        if image not in images:
            images.append(image)
        captions.append(caption)

    return images, captions


def create_image_captions_map(images, captions):
    image_captions_map = {}
    counter = 0
    step = 5
    for image in images:
        image_captions_map[image] = captions[counter:counter + step]
        counter += step
    return image_captions_map


def read_and_preprocess_image(base_dir, filename):
    image_path = base_dir + filename
    image = Image.open(image_path)

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_tensor = preprocess(image)
    return image_tensor


def clean_caption(caption):
    punctuations = string.punctuation
    # punctuations = punctuations.replace(".", "")
    caption = caption.translate(str.maketrans("", "", punctuations))
    caption = caption.casefold()
    return caption


def tokenizer(sentence):
    list_of_words = word_tokenize(sentence)
    return list_of_words


def create_word_ix_map_and_vocabulary(list_of_words):
    word_to_ix_map = {}
    vocabulary = []
    for word in list_of_words:
        if word not in word_to_ix_map:
            word_to_ix_map[word] = len(word_to_ix_map)
            vocabulary.append(word)
    return word_to_ix_map, vocabulary


def train_test_split(images_list, captions_list, train_size):
    index = int(train_size * len(images_list))
    train_images, test_images = images_list[:index], images_list[index:]
    train_caption, test_caption = captions_list[:index], captions_list[index:]
    return train_images, test_images, train_caption, test_caption


# text = read_caption_file(filepath)
# image_files, captions = extract_images_and_caption(text)
# image_caption_map = create_image_captions_map(image_files, captions)








