from collections import Counter

UNK_TOKEN = "<unk>"


class Vocab:
    """
    Vocabulary Class
    Store the index mapping for the tokens and recognize the unknown token and then return it
    """

    def __init__(self, tokens, base_map={}, max_size=None, least_freq=0):
        self.token2idx = base_map
        # count the word/token/tags frequency
        self.freq = Counter(
            [token for sequence in tokens for token in sequence]
        )

        vocab_size = 0
        # store the token start from higher frequency
        for word, count in sorted(
            self.freq.items(), key=lambda item: item[1], reverse=True
        ):
            if count < least_freq:
                break
            # if vocab size is larger than max size, stop inserting words into vocab
            if max_size is not None and vocab_size > max_size:
                break
            self.insert(word)
            vocab_size += 1

        self.idx2token = reverse_map(self.token2idx)

    def insert(self, token):
        if token in self.token2idx.keys():
            return
        self.token2idx[token] = len(self.token2idx)

    def lookup_index(self, word):
        if word not in self.token2idx.keys():
            word = UNK_TOKEN
        return self.token2idx[word]

    def lookup_token(self, idx):
        return self.idx2token[idx]

    def __len__(self):
        return len(self.token2idx)

    def __repr__(self):
        return str(self.token2idx)


def reverse_map(_map):
    reversed_map = {}
    for key, val in _map.items():
        reversed_map[val] = key
    return reversed_map
