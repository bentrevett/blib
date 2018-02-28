from collections import Counter
from tqdm import tqdm

class Vocab:

    def __init__(self, max_size=None, min_freq=0, max_length=None, pad=False, unk_token='<UNK>', pad_token='<PAD>', start_token=None, end_token=None, tokenizer=None):

        self.max_size = max_size
        self.min_freq = min_freq
        self.max_length = max_length
        self.pad = pad
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.start_token = start_token
        self.end_token = end_token
        self.tokenizer = tokenizer

        self.next_id = 0
        self.token_to_id = {}
        self.id_to_token = {}

        assert self.min_freq >= 0, "min_freq must be 0 or greater"

        if self.max_size is not None and self.min_freq>0:
            assert self.unk_token is not None, "If unk_token = None, then you cannot set max_size or min_freq"

        if self.pad:
            assert self.max_length is not None, 'If you want to pad all sequences to a certain length, need to specify a max. length to pad to'
            assert self.pad_token is not None, 'If you want to pad, cannot have pad_token == None'

        if self.tokenizer == None:
            #default tokenizer is to split the string on spaces
            self.tokenizer = lambda x : x.split(' ')
        elif self.tokenizer == 'chars':
            #splits string into individual characters
            self.tokenizer = lambda x : list(x)
        elif self.tokenizer == 'spacy':
            #wrapper for the spacy tokenizer
            import spacy
            nlp = spacy.load('en')
            self.tokenizer = lambda x : [token.text for token in nlp.tokenizer(x)]
        else:
            #if you specify your own tokenizer, assert it is callable
            assert callable(self.tokenizer), 'Supplied tokenizer must be a callable function/method'

        if self.pad_token is not None:
            assert type(self.pad_token) is str, 'pad_token must be a string'
            self.add_or_get_id(self.pad_token)
        if self.unk_token is not None:
            assert type(self.unk_token) is str, 'unk_token must be a string'
            self.add_or_get_id(self.unk_token)
        if self.start_token is not None:
            assert type(self.start_token) is str, 'start_token must be a string'
            self.add_or_get_id(self.start_token)
        if self.end_token is not None:
            assert type(self.end_token) is str, 'end_token must be a string'
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
                    if self.max_length is None:
                        counter.update(tokens)
                    else:
                        counter.update(tokens[:self.max_length]) #TODO: this is inefficient 

        if self.max_size == None:
            #if you don't set a max_size then nothing is unk'd
            self.max_size = len(counter)

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
                if self.max_length is None:
                    tokens = self.tokenizer(text)
                else:
                    tokens = self.tokenizer(text) #TODO: this is inefficient 
                    tokens = tokens[:self.max_length]
                if self.start_token is not None:
                    tokens = [self.start_token] + tokens
                if self.end_token is not None:
                    tokens = tokens + [self.end_token]
                if self.pad:
                    while(len(tokens)<self.max_length):
                        tokens.append(self.pad_token)
                _temp.append(self.get_id_or_unk(tokens))
            temp.append(_temp)

        return temp

def build_vocab(sources, max_size=None, min_freq=0, max_length=None, pad=False, unk_token='<UNK>', pad_token='<PAD>', start_token=None, end_token=None, tokenizer=None):
    """
    Does the same thing as: 
    vocab = blib.text.Vocab(...)
    vocab.build_vocab(sources, ...)
    """
    
    vocab = Vocab(max_size, min_freq, max_length, pad, unk_token, pad_token, start_token, end_token, tokenizer)

    vocab.build_vocab(sources)

    return vocab

def tokenize(vocab, sources):
    """
    Does the same thing as:
    tokenized_X = vocab.tokenize(X)
    """

    return vocab.tokenize(sources)

def build_and_tokenize(build_sources, tokenize_sources=None, max_size=None, min_freq=0, max_length=None, pad=False, unk_token='<UNK>', pad_token='<PAD>', start_token=None, end_token=None, tokenizer=None):
    """
    This does the building of the vocab and the tokenization in a single step
    build_sources is what you want to use to build the vocab, i.e. can build from just train or train, val and test.
    tokenize_sources are the sources you want to tokenize (99% of the time should be train, val and test)
    """
    if tokenize_sources == None:
        tokenize_sources = build_sources

    vocab = Vocab(max_size, min_freq, max_length, pad, unk_token, pad_token, start_token, end_token, tokenizer)

    vocab.build_vocab(build_sources)

    #need to wrap in tuple to unpack
    return (vocab, *vocab.tokenize(tokenize_sources))

    
