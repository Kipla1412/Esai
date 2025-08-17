import re 
import string
import regex

from ..base import Pipeline

class Tokenizer(Pipeline):

    STOP_WORDS =  {"a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it",
                  "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these",
                  "they", "this", "to", "was", "will", "with"}
    
    @staticmethod
    def tokenize(text,lowercase =True,emoji =True,alphanum = True,stopwords =True):

        return Tokenizer(lowercase,emoji,alphanum,stopwords)
    
    def __init__(self,lowercase =True, emoji =True, alphanum =False,stopwords =False):

        self.lowercase = lowercase

        self.alphanum,self.segment = None,None

        if self.alphanum:
            self.alphanum = re.compile(r"^\d*[a-z][\-.0-9:_a-z]{1,}$")

        #text segmentation and  per Unicode Standard Annex
        else:
            pattern = r"\w\p{Extended_Pictographic}\p{WB:RegionalIndicator}" if emoji else r"\w"
            self.segment = regex.compile(rf"[{pattern}](?:\B\S)*", flags=regex.WORD)
        self.stopwords = stopwords if isinstance(stopwords,str) else Tokenizer.STOP_WORDS if stopwords else False

    def __call__(self,text):

        if text is None:
            return None

        text = text.lower() if self.lowercase else text

        if self.alphanum:
            tokens =[token.strip(string.punctuation) for token in text.split()] 
            tokens =[token for token in tokens if re.match(self.alphanum,token)]

        else:

            tokens = regex.findall(self.segment,text)

        if self.stopwords:

            tokens = [token for token in tokens if token not in self.stopwords]

        return tokens      