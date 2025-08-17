
import os
import sys
from array import array
import sqlite3

import numpy as np
from collections import Counter
from threading import RLock
import functools

class Terms:

    """
    Indexing,searching
    """
    CREATE_TERMS ="""
    
     CREATE TABLE IF NOT EXISTS terms(
     terms TEXT PRIMARY KEY,
     ids BLOB,
     freq BLOB
     )
    """
    INSERT_TERM = "INSERT OR REPLACE INTO terms VALUES (?, ?, ?)"
    SELECT_TERMS = "SELECT ids, freqs FROM terms WHERE term = ?"


    CREATE_DOCUMENTS ="""
    
    CREATE TABLE IF NOT EXISTS documents(
    indexid INTEGER PRIMARY KEY,
    id TEXT,
    deleted INTEGER,
    length INTEGER
    )
    """
    DELETE_DOCUMENTS = "DELETE FROM documents"
    INSERT_DOCUMENT ="INSERT OR REPLACE INTO documents VALUES(?, ?, ?)"
    SELECT_DOCUMENTS = "SELECT indexid, id, deleted, length FROM documents ORDER BY indexid"

    def __init__(self,config,score,idf):

        self.config = config if isinstance(config, dict) else {}
        self.cachelimit = self.config.get("cachelimit",2500000000)
        self.cutoff = self.config.get("cutoff",0.1)

        #scoring

        self.score,self.idf = score,idf
        # document ids

        self.ids ,self.deletes,self.lengths =[],[],array("q")
        # terms cache

        self.terms,self.cachesize ={},0
        # terms database

        self.connection,self.cursor,self.path = None,None,None
        self.lock = RLock()

    def insert(self,uid,terms):

        self.intialize()

        indexid = len(self.ids)

        freqs,length = Counter(terms), len(terms)

        for term,count in freqs.items():

            self.add(indexid,term,count)
        
        self.cachesize += 16

        if self.cachesize >= self.cachelimits:
            self.index()

        self.ids.append(uid)
        self.lengths.append(length)


    def intialize(self):

        if not self.connection:

            self.connection = self.connect()
            self.cursor = self.connection.cursor()

            self.cursor.execute(Terms.CREATE_TERMS)
            self.cursor.execute(Terms.CREATE_DOCUMENTS)

    def add(self,indexid,term,freq):

        if term  not in self.terms:
            self.terms[term] = (array("q"),array("q"))

        ids,freqs = self.terms[term]

        ids.append(indexid)
        freqs.append(freq)

    def index(self,uid,tokens):
        
        for term,(nuids, nfreqs) in self.terms.items():
            uids,freqs = self.lookup(term)

            if uids:
                uids.extend(nuids)
                freqs.extend(nfreqs)

            else:
                uids, freqs = nuids,nfreqs

            if sys.byteorder == "big":
                uids.byteswap()
                freqs.byteswap()

            result = self.cursor.execute(Terms.INSERT_TERM,[term, uids.tobytes(), freqs.tobytes()])

        self.weights.cache_clear()

        self.terms, self.cachesize = {},0

    def lookup(self,term):#lookup() → gets raw term positions & counts

        uids, freqs = None, None
        result = self.cursor.execute(Terms.SELECT_TERMS,[term]).fetchone()

        if result :

            uids, freqs =(array("q"), array("q"))
            uids.frombytes(result[0])
            freqs.frombytes(result[1])

            if sys.byteorder == "big":

                uids.byteswap()
                freqs.byteswap()
        return uids,freqs
    
    @functools.lru_cache(maxsize=500)
    def weights(self,term):#weights() → converts them to search scores using TF-IDF or BM25

        lengths = np.frombuffer(self.lengths, dtype =np.int64)

        with self.lock:

            uids, freqs = self.lookup(term)
            weights =None

            if uids:

                uids = np.frombuffer(uids ,dtype = np.int64)

                weights = self.score(
                    np.frombuffer(freqs, dtype = np.int64),
                    self.idf[term], lengths[uids]
                ).astype(np.float32)
            
            return uids,weights
        
    def delete(self,ids):                      

        self.deletes.extend([self.ids.index(i) for i in ids])
        # self.ids = ["doc1","doc2","doc3"]
        # self.ids.index("document") 

    def search(self,query,limit):

        scores = np.zeros(len(self.ids), dtype= np.float32)

        terms,skipped,hasscore = Counter(terms),{},False
        for term, freq in terms.items():

            uids,weights = self.weights(term)#--> its return uids and weights --> doc freq,idf and document lengths like scoring

            if uids is not None:
                
                if len(uids) <= self.cutoff * len(self.ids):

                    scores[uids] += freq * weights

                    hasscore = True # uncommon word to create score

                else:
                    skipped = [term] # common word used for future merging

            return self.topn(scores,limit,hasscore,skipped)
        
    def topn(self,scores,limit,hasscore,skipped):

        topn = min(len(self.ids), limit* 5)
        matches = self.candidates(scores,topn)

        self.merge(scores,matches,hasscore,skipped)

        if not hasscore:

            matches = self.candidates(scores,topn)

        matches = matches[np.argsort(-scores[matches])]

        return [(self.ids[x], float(scores[x])) for x in matches[:limit] if scores > 0]
    
    def candidates(self,scores,topn):

        scores[self.deletes] = 0
        return np.argpartition(scores, -topn)[-topn:]
    
    def merge(self,scores,matches,hasscore,skipped):
        
        for term, freq in self.terms.items():

            uids,weights = self.weights(term)

            if hasscore:

                indices = np.searchsorted(uids,matches)

                indices = [x for i,x in enumerate(indices) if x < len(uids) and uids[x] ==  matches[i]]

                uids, weights = uids[indices], weights[indices]

            scores[uids] += freq * weights    