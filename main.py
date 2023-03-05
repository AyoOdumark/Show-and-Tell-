from preprocessing import read_caption_file, read_and_preprocess_image, create_image_captions_map, create_word_ix_map_and_vocabulary
from preprocessing import train_test_split, clean_caption, tokenizer, extract_images_and_caption
from torchvision.models import vgg16, VGG16_Weights
from models import ShowAndTellModel
import torch

TRAIN_SIZE = 0.75


def loss_function(list_of_losses):
    total = 0.0
    for loss in list_of_losses:
        total += loss
    return total


# Read file
PATH = ".\\captions.txt"
BASE_DIR = ".\\Images\\"
caption_text = read_caption_file(path=PATH)

# Extract images and captions and create an image-caption map
images_path, captions = extract_images_and_caption(caption_text)
image_caption_map = create_image_captions_map(images_path, captions)


# Preprocessing of captions
def normalize_and_tokenize(preprocessor, tokenizer, captions):
    for key in captions:
        captions[key] = [preprocessor(caption) for caption in captions[key]]
        captions[key] = [tokenizer(caption) for caption in captions[key]]
    return captions


image_caption_map = normalize_and_tokenize(clean_caption, tokenizer, image_caption_map)

# create word-index map and vocabulary and a caption list
captions_list = []
words = []
for key in image_caption_map:
    captions_list.append(image_caption_map[key])
    for sentence in image_caption_map[key]:
        words += sentence

word_ix_map, vocabulary = create_word_ix_map_and_vocabulary(words)


# Read and preprocess images
image_tensor_list = []
for image_file in image_caption_map:
    image = read_and_preprocess_image(BASE_DIR, image_file)
    image_tensor_list.append(image)

# create training data and test data
train_images, test_images, train_captions, test_captions = train_test_split(image_tensor_list, captions_list, TRAIN_SIZE)

# HYPERPARAMETERS
NUM_EPOCHS = 10
HIDDEN_SIZE = 1000
EMBEDDING_DIM = 4096
VOCAB_SIZE = len(vocabulary)
NUM_ITERATIONS = len(train_images)

# Using vggnet as feature extractor
vggnet = vgg16(VGG16_Weights.DEFAULT)

# instantiate our model
model = ShowAndTellModel(vggnet, EMBEDDING_DIM, VOCAB_SIZE, HIDDEN_SIZE)

# Define optimizer and loss function
criterion = torch.nn.NLLLoss()
Adam = torch.optim.Adam(model.parameters())

# Training_loop
for epoch in range(NUM_EPOCHS):
    for image, captions in zip(train_images, train_captions):
        running_loss = []

        # forward pass
        for caption in captions:
            losses = []
            hidden_state, cell_state = model.decoder.init_hidden()
            hidden_state, cell_state = model.forward(image, hidden_state, cell_state, True)
            for i in range(len(caption)):
                if i+1 < len(caption):
                    word_idx = torch.LongTensor([word_ix_map[caption[i]]])
                    next_word_idx = torch.LongTensor([word_ix_map[caption[i+1]]])
                    output, hidden_state, cell_state = model.forward(word_idx, hidden_state, cell_state, False)
                    losses.append(criterion(output, next_word_idx))

            # compute gradients and update weights
            loss = loss_function(losses) / len(losses)
            loss.backward()
            Adam.step()

            # Prevent weight accumulation
            Adam.zero_grad()

            running_loss.append(loss.item())

        print(f"Loss: {running_loss}")











