import torch.nn as nn
import torch


class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        model.features.requires_grad_(False)  # Hold the Convet fixed and use as a feature extractor
        model.classifier.requires_grad_(False)
        self.features = nn.Sequential(*list(model.features))
        self.pool = model.avgpool
        self.flatten = nn.Flatten(start_dim=0)
        self.classifier = nn.Sequential(
            model.classifier[0],
            model.classifier[3]
        )

    def forward(self, image):
        output = self.features(image)
        output = self.pool(output)
        output = self.flatten(output)
        output = self.classifier(output)
        return output


class LSTMDecoder(nn.Module):
    def __init__(self, embedding_dim, vocab_size, hidden_size):
        super(LSTMDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstmcell = nn.LSTMCell(input_size=embedding_dim, hidden_size=hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_tensor, hidden_state, cell_state, is_image=False):
        if is_image:
            feature_vector = input_tensor.unsqueeze(0)
            hidden_state, cell_state = self.lstmcell(feature_vector, (hidden_state, cell_state))
            return hidden_state, cell_state
        else:
            embed = self.embedding(input_tensor)
            # embed = embed.unsqueeze(1)
            # combined = torch.cat((feature_vector.unsqueeze(0), embeds), dim=0)
            # combined = combined.unsqueeze(1)
            hidden_state, cell_state = self.lstmcell(embed, (hidden_state, cell_state))
            output = torch.log_softmax(self.linear(hidden_state), dim=1)
            return output, hidden_state, cell_state

    def init_hidden(self):
        hidden_state, cell_state = torch.zeros((1, self.hidden_size)), torch.zeros((1, self.hidden_size))
        return hidden_state, cell_state


class ShowAndTellModel(nn.Module):
    def __init__(self, image_model, embedding_dim, vocab_size, hidden_size):
        super(ShowAndTellModel, self).__init__()
        self.extractor = FeatureExtractor(image_model)
        self.decoder = LSTMDecoder(embedding_dim, vocab_size, hidden_size)

    def forward(self, input_tensor, hidden_state, cell_state, is_image=False):
        if is_image:
            feature_vector = self.extractor(input_tensor)
            hidden_state, cell_state = self.decoder(feature_vector, hidden_state, cell_state, is_image=is_image)
            return hidden_state, cell_state
        else:
            output, hidden_state, cell_state = self.decoder(input_tensor, hidden_state, cell_state, is_image=is_image)
            return output, hidden_state, cell_state




