from nltk.corpus import state_union as su
from nltk.tokenize import PunktSentenceTokenizer, word_tokenize
from nltk import pos_tag, RegexpParser


train_text = su.raw("2005-GWBush.txt")  # training our tokenizer using 2005 speech of GW Bush
sample_text = su.raw("2006-GWBush.txt")  # testing our tokenizer model on 2006 speech of GW Bush
custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
tokenized = custom_sent_tokenizer.tokenize(sample_text)


def process_content():
    try:
        for i in tokenized:
            words = word_tokenize(i)
            tagged = pos_tag(words)
            chunkGram = r"""Chunk: {<.*>+} }<VB.?|IN|DT>+{"""  # chunking and chinking of our data
            chunkParser =RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            print(chunked)
            chunked.draw()
    except Exception as e:
        print(str(e))


process_content()
