import torch


class Word2Vec(torch.nn.Module):
    def __init__(self, arch, voc, emb):
        super().__init__()
        self.embeddings = torch.nn.Embedding(num_embeddings=voc, embedding_dim=emb)
        self.linear = torch.nn.Linear(in_features=emb, out_features=voc, bias=False)
        self.sig = torch.nn.Sigmoid()

        if arch in ["cbow", "skipgram"]:
            self.arch = arch
        else:
            raise ValueError(f"Unknown architecture: {arch}")

    def forward(self, center, context, rand):
        input = center if self.arch == "skipgram" else context
        output = context if self.arch == "skipgram" else center

        input_embeddings = self.embeddings(input)
        output_weights = self.linear.weight[output]
        random_weights = self.linear.weight[rand]

        if self.arch == "skipgram":
            input_embeddings = input_embeddings.unsqueeze(-1)
        else:
            output_weights = output_weights.unsqueeze(1)
            random_weights = random_weights.unsqueeze(1)
            input_embeddings = input_embeddings.permute(0, 2, 1)

        dot_product = torch.bmm(output_weights, input_embeddings).squeeze()
        random_dot_product = torch.bmm(random_weights, input_embeddings).squeeze()

        activation = self.sig(dot_product)
        random_activation = self.sig(random_dot_product)

        positive_sampling_loss = -activation.log().mean()
        negative_sampling_loss = -(1 - random_activation + 10 ** (-3)).log().mean()
        return positive_sampling_loss + negative_sampling_loss
