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

        # TODO: make this better

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

        # TODO: handle unknown words

        return words

    def text_to_token_ids(self, text):
        words = self.text_to_tokens(text)
        tokens = [self.vocab_mapping[word] for word in words]
        return tokens

    def token_ids_to_text(self, token_ids):
        words = [self.inv_vocab_mapping[token_id] for token_id in token_ids]
        text = " ".join(words)
        return text


if __name__ == "__main__":
    corpus = ["There are a lot of words in this corpus."]
    text = "In this corpus there are words."

    def test_tokeniser(tokeniser):
        tokens = tokeniser.text_to_tokens(text)
        token_ids = tokeniser.text_to_token_ids(text)
        reconstructed_text = tokeniser.token_ids_to_text(token_ids)
        print("Tokens:")
        print(tokens)
        print("Token IDs:")
        print(token_ids)
        print("Reconstructed text:")
        print(reconstructed_text)

    new_tokeniser = Tokeniser(corpus=corpus)
    restored_tokeniser = Tokeniser()

    test_tokeniser(new_tokeniser)
    print("-" * 50)
    test_tokeniser(restored_tokeniser)
