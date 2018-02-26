from collections import Counter
from tqdm import tqdm

class Vocab:

    def __init__(self, max_size=1_000_000, min_freq=0, unk_token='<UNK>', pad_token='<PAD>', start_token=None, end_token=None, tokenizer=None):

        self.max_size = max_size
        self.min_freq = min_freq
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.start_token = start_token
        self.end_token = end_token
        self.tokenizer = tokenizer

        if self.tokenizer == None:
            self.tokenizer = lambda x : x.split(' ')

        if self.tokenizer == 'spacy':
            import spacy
            nlp = spacy.load('en')
            self.tokenizer = lambda x : [token.text for token in nlp.tokenizer(x)]

        self.next_id = 0
        self.token_to_id = {}
        self.id_to_token = {}

        if pad_token is not None:
            self.add_or_get_id(self.pad_token)
        if unk_token is not None:
            self.add_or_get_id(self.unk_token)
        if start_token is not None:
            self.add_or_get_id(self.start_token)
        if end_token is not None:
            self.add_or_get_id(self.end_token)

    def __len__(self):
        return len(self.token_to_id)

    def add_or_get_id(self, token):
        """
        input a token, if it exists in the dictionary already then return the id
        if it doesn't exist in the dictionary, then it adds it and returns the id
        """
        if token in self.token_to_id:
            return self.token_to_id[token]

        this_id = self.next_id
        self.next_id += 1
        self.token_to_id[token] = this_id
        self.id_to_token[this_id] = token

        return this_id

    def get_id_or_unk(self, tokens):
        """
        input a list of tokens, get the id if it exists in the dictionary, get the <UNK> token if it doesn't
        """
        ids = []

        for token in tokens:
            if token in self.token_to_id:
                ids.append(self.token_to_id[token])
            else:
                ids.append(self.token_to_id[self.unk_token])

        return ids

    def get_name_for_id(self, token_id):
        """
        go from id back to token
        """
        return self.id_to_token[token_id]

    def build_vocab(self, sources):
        """
        Given a source (a list containing a list of strings): tokenize to split the strings into lists where each element is a token and then build a counter of the tokens.
        Then use this counter to build the vocabulary.
        """

        counter = Counter()

        #sources is tuple of lists of lists of strings
        for source in sources:
            #source is a list of list of strings
            for text in tqdm(source, desc='Building counter'):
                    tokens = self.tokenizer(text)
                    counter.update(tokens)

        for token, count in tqdm(counter.most_common(self.max_size), desc='Building dictionary'):
            if count >= self.min_freq:
                self.add_or_get_id(token)

    def tokenize(self, sources):

        temp = []

        #sources is a tuple of list of list of strings
        for source in sources:
            _temp = []
            #source is a list of strings
            for text in tqdm(source, desc='Tokenizing'):
                #text is a string
                tokens = self.tokenizer(text)
                if self.start_token is not None:
                    tokens = [self.start_token] + tokens
                if self.end_token is not None:
                    tokens = tokens + [self.end_token]
                _temp.append(self.get_id_or_unk(tokens))
            temp.append(_temp)

        return temp

def build_vocab(sources, max_size=1_000_000, min_freq=0, unk_token='<UNK>', pad_token='<PAD>', start_token=None, end_token=None, tokenizer=None):
    """
    Does the same thing as: 
    vocab = blib.text.Vocab(...)
    vocab.build_vocab(sources, ...)
    """
    
    vocab = Vocab(max_size, min_freq, unk_token, pad_token, start_token, end_token, tokenizer)

    vocab.build_vocab(sources)

    return vocab

def tokenize(vocab, sources):
    """
    Does the same thing as:
    tokenized_X = vocab.tokenize(X)
    """

    return vocab.tokenize(sources)

def build_and_tokenize(sources, max_size=1_000_000, min_freq=0, unk_token='<UNK>', pad_token='<PAD>', start_token=None, end_token=None, tokenizer=None):
    """
    If you want to build your vocab on all of the train/val/test set, then this is a shorthand way of doing it.
    If you only want to build your vocab on the train/val set and not the test set (or something similar), then DON'T USE THIS.
    """

    vocab = Vocab(max_size, min_freq, unk_token, pad_token, start_token, end_token, tokenizer)

    vocab.build_vocab(sources)

    return vocab.tokenize(sources)

    
