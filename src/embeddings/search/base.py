import logging

from .errors import IndexNotFoundError

class Search:

    def __init__(self,embeddings,indexids =False, indexonly =False):

        self.embeddings =embeddings
        self.indexids =indexids or indexonly
        self.indexonly = indexonly

        #alias embeddingg attributes

        self.ann = embeddings.ann
        self.batchtransform = embeddings.batchtransform
        self.indexes = embeddings.indexes
        self.ids =embeddings.ids

        self.query = embeddings.query
        self.scoring =embeddings.scoring if embeddings.issparse() else None

    def __call__(self,queries,limit =None,weights =None,index =None,parameters =None):

        limit = limit if limit else 3
        weights = weights if weights is not None else 0.5

        if not index and not self.ann and not self.scoring and not self.indexes and not self.database:
            return [[]] * len(queries)
        
        if not index and not self.ann and not self.scoring and self.indexes:

            index = self.indexes.default()

        return self.search(queries,limit,weights,index,parameters)
    
    def search(self,queries,limit,weights,index,parameters):

        if index:
            return self.subindex(queries,limit,weights,index)
        
        hybrid = self.ann and self.scoring

        dense = self.dense(queries , limit*10 if hybrid else limit) if self.ann else None
        sparse= self.sparse(queries, limit*10 if hybrid else limit) if self.scoring else None

        if hybrid:

            if isinstance(weights,(int,float)):
                weights =[weights, 1-weights]

            results =[]

            for vectors in zip(dense,sparse):

                for v,scores in enumerate(vectors):

                    for r,(uid,score) in enumerate(scores if weights[v] > 0 else []):

                        uids ={}
                        if uid not in uids:

                            uids[uid] =0.0

                        if self.scoring.isnormalized():

                            uids[uid] += score * weights[v] # convex comnination

                        else :

                            uids[uid] += (1.0/(r+1)) * weights[v] # reciprocal rank fusion

                results.append(sorted(uids.items(), key = lambda x:x[1], reverse= True [:limit]))

            return results 
        
        if not sparse and not dense:
            raise IndexNotFoundError("no indexes available")
        
        return dense if dense else sparse
    

    def subindex(self, queries,limit,weights,index):

        if not self.indexes or index not in self.indexes:

            raise IndexNotFoundError(f"index '{index}' not available")
        
        results = self.indexes[index].batchsearch(queries,limit,weights)
        return self.resolve(results)
    
    def resolve(self,results):

        if not self.indexids and self.ids:

            return[[(self.ids[i],score) for i, score in r ]for r in results]
        
        return results
    
    def dense(self,queries,limit):

        embeddings = self.batchtransform((None,query,None)for query in queries) 

        results = self.ann.search(embeddings,limit)
        results = [[(i,score) for i,score in r if score > 0]for r in results ]

        return self.resolve(results)
    
    def sparse(self, queries, limit):

        results = self.scoring.batchsearch(queries, limit)
        results = [[(i,score) for i, score in r if score > 0] for r in results]

        return self.resolve(results)
    
    def limit(self,queries):

        qlimit = 0

        for query in queries:

            l = query.get("limit")
            if l and l.isdigit():
                l = int(l)

            qlimit = l if l and l > qlimit else qlimit

            return qlimit








        

