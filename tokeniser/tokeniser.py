import os
import re


class Tokeniser:
    def __init__(self, corpus=None):
        script_dir = os.path.dirname(__file__)
        self.vocab_mapping_path = os.path.join(script_dir, "vocab_mapping.txt")
        self.vocab_mapping = None
        self.inv_vocab_mapping = None
        self.pattern = re.compile(r"\b\w+\b")
        self._initialise_vocab_mapping(corpus)

    def _initialise_vocab_mapping(self, corpus):
        # if corpus is provided then we want to generate a new vocabulary
        if corpus:
            self.vocab_mapping = self._generate_vocab_mapping(corpus)
        # if corpus is not provided then use the pre-existing vocabulary
        else:
            self.vocab_mapping = self._read_vocab_mapping()

        print("Vocabulary mapping:")
        print(self.vocab_mapping)

        # generate the inverse mapping
        self.inv_vocab_mapping = {idx: word for word, idx in self.vocab_mapping.items()}

        print("Inverse vocabulary mapping:")
        print(self.inv_vocab_mapping)

    def _generate_vocab_mapping(self, corpus):
        vocab = set()
        for text in corpus:
            words = self.text_to_tokens(text)
            vocab.update(words)
        vocab_mapping = {word: idx for idx, word in enumerate(vocab)}

        self._save_vocab_mapping(vocab_mapping)

        return vocab_mapping

    def _save_vocab_mapping(self, vocab_mapping):
        with open(self.vocab_mapping_path, "w") as f:
            for word in vocab_mapping:
                f.write(f"{word}\n")

    def _read_vocab_mapping(self):
        vocab_mapping = {}
        with open(self.vocab_mapping_path, "r") as f:
            for idx, word in enumerate(f):
                vocab_mapping[word.strip()] = idx
        return vocab_mapping

    def text_to_tokens(self, text):
        text = text.lower()
        words = self.pattern.findall(text)

        # TODO: make this better!

        return words

    def text_to_token_ids(self, text):
        words = self.text_to_tokens(text)
        tokens = [self.vocab_mapping[word] for word in words]
        return tokens


if __name__ == "__main__":
    corpus = ["This is a test.", "This is another test."]
    text = "This is test."

    tokeniser = Tokeniser(corpus=corpus)
    tokens = tokeniser.text_to_tokens(text)
    token_ids = tokeniser.text_to_token_ids(text)
    print("Tokens:")
    print(tokens)
    print("Token IDs:")
    print(token_ids)

    print("-" * 50)

    tokeniser = Tokeniser()
    tokens = tokeniser.text_to_tokens(text)
    token_ids = tokeniser.text_to_token_ids(text)
    print("Tokens:")
    print(tokens)
    print("Token IDs:")
    print(token_ids)
