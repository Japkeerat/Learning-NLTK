from nltk.corpus import state_union as su
from nltk.tokenize import PunktSentenceTokenizer, word_tokenize
from nltk import pos_tag, ne_chunk


train_text = su.raw("2005-GWBush.txt")  # using 2005 speech of GWBush for training our tokenizer model
sample_text = su.raw("2006-GWBush.txt")  # applying the model on 2006 sppech of GWBush
custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
tokenized = custom_sent_tokenizer.tokenize(sample_text)


# This method is used to tag words.
def process_content():
    try:
        for i in tokenized:
            words = word_tokenize(i)
            tagged = pos_tag(words)
            named = ne_chunk(tagged)
            print(named)
    except Exception as e:
        print(str(e))


process_content()
