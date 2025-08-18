from ..base import Pipeline
from ..data import Tokenizer
from ..llm import LLM
from .factory import GenerationFactory
from ..text import Questions
from ..text import Similarity
from ...models import Models


class RAG(Pipeline):

    def __init__(self,similarity,path,quantize = False,gpu =True,model =None,tokenizer =None,minscore = None,mintokens =None,
                 context = None,task = None,output ="default",template = None, separator = " ",system = None, **kwargs):
        
        self.similarity = similarity
        self.model = self.load(path,quantize,gpu,model,task,**kwargs)
        self.tokenizer = tokenizer if tokenizer else Tokenizer() if hasattr(self.similarity,"scoring") and self.similarity.isweighted() else None

        self.minscore = minscore if self.minscore is not None else 0.0
        self.mintokens = mintokens if self.mintokens is not None else 0.0

        self.context = context if self.context else 3
        self.output = output
        self.template = template if self.template else "{question} {context}"

        self.separator = separator
        self.system = system

    def __call__(self,queue,texts = None, **kwargs):

        inputs = queue

        queue= queue if isinstance(queue,list) else [queue]

        if queue and isinstance(queue[0],dict):

            queue = [tuple(row.get(x) for x in ["name","query","question","snippet"])  for row in queue]

        if queue and isinstance(queue[0],str):

            queue =[(None,row,row,None)for row in queue]

        results =self.query( [query for _,query,_,_ in queue], texts)

        names, queries,questions,contexts,topns,snippets = [],[],[],[],[],[]

        for x ,[name,query,question,snippet] in enumerate(queue):

            topn = sorted(results[x], key =lambda y:y[2],reverse= True)[:self.context]

            context = self.separator.join(text for _,text,_ in(sorted(topn, key =lambda y:y[0]) if texts else topn))

            names.append(name)
            queries.append(query)
            questions.append(question)
            contexts.append(context)
            topns.append(topn)
            snippets.append(snippet)

        answers = self.answers(questions,contexts,**kwargs)

        return self.apply(inputs, names, queries, answers, topns, snippets) if isinstance(answers,list) else answers
    
    def load(self,path,quantize,gpu,model,task,*kwargs):

        if not isinstance(path,str):

            return path
        
        task =GenerationFactory.method(path,task)
        task = Models.task(path,**kwargs) if task == "transformers" else task

        if task == "question-answering":

            return Questions(path,quantize,gpu,model,**kwargs)
        
        return LLM(path=path,quantize=quantize,gpu=gpu,model=model,task=task,**kwargs)
    
    def query(self,queries,texts):

        if not queries:

            return []
        
        scores, segments,tokenlist = self.score(queries,texts)
        results =[]

        for i,query in enumerate(queries):

            must = [token.strip("+") for token in query.split() if token.startswith("+") and len(token) > 1]
            mnot =[token.strip("-") for token in query.split() if token.startswith("-") and len(token) > 1]

            segment = segments if texts else segment[i]
            tokens = tokenlist if texts else tokenlist[i]

            # matches the query and context score

            matches =[]

            for y,(x,score) in enumerate(scores[i]):

                x = x if texts else y
                text = segment[x][1] # its looks like [(0,"hello kipla")]

                if (not must or all(token.lower() in text.lower() for token in must)) and (
                    not mnot or all(token.lower() not in text.lower() for token in mnot)):

                    if score >= self.minscore and len(tokens[x]) >= self.mintokens :

                        matches.append(segment[x],(score,))

            results.append(matches)

        return results
    
    def score(self,queries,texts):

        scores,segments,tokenlist =[],[],[]

        if texts:

            for text in texts:

                tokens = self.tokenize(text)

                if tokens:
                    segments.append(text)
                    tokenlist.append(tokens)

                segments = list(enumerate(segments))

        if isinstance(self.similarity, Similarity):

            scores = self.similarity(queries,[t for _,t,_ in segments])
        elif texts:

            scores = self.similarity.batchsimilarity([self.tokenize[x] for x in queries], tokenlist)
        else:
            scores,segments,tokenlist = self.batchsearch(queries)
        
        return scores,segments,tokenlist
    
    def batchsearch(self,queries):

        scores,segments, tokenlist = [],[],[]

        for results in self.similarity.batchsearch([self.tokenize[x] for x in queries], tokenlist):

            scores.append([(result["id"],result["score"]) for result in results])
            segments.append([( result["id"], result["text"])for result in results])
            tokenlist.append([self.tokenize(result ["text"]) for result in results])

        return scores, segments,tokenlist
    
    def tokenize(self,text):

        return self.tokenizer(text) if self.tokenizer else text
    
    def answers(self,questions,contexts,**kwargs):

        if isinstance(self.model, Questions):

            return self.model(questions,contexts)
        
        return self.pipeline(self.prompts(questions,contexts),**kwargs)
    
    def prompts(self,questions,contexts):

        prompts =[]

        for x,context in enumerate(contexts):

            prompt = self.template.format( question = questions[x], context =contexts[x])

            if self.system:

                prompt =[

                    {"role":"system", "content": self.system.format(question=questions[x], context =contexts)},
                    {"role":"user","content": prompt},
                ]
            prompts.append(prompt)

        return prompts
    
    def apply(self,inputs,names,queries,answers,topns,snippets):

        answers = self.snippets(names,queries,topns,snippets)
        
        if self.output == "flatten":

            answers =[answer for _,answer in answers]

        else:
            
            if self.output == "reference":

                answers =self.reference(queries,answers,topns)
            
            first = inputs[0] if inputs and isinstance(answers,list) else inputs

            if isinstance(first,(dict,str)):

                fields = ["name","answer","reference"] if isinstance(first,dict) and "name" in first else [None, "answers","reference"]

                answers =[{fields[x] : column for x, column in enumerate(row) if fields[x]}for row in answers]

        return answers[0] if answers and  isinstance(inputs,(tuple,dict,str)) else answers 
    
    def snippets(self,names,answers,topns,snipetts):

        results =[]

        for x, answer in enumerate(answers):

            if answer and snipetts[x]:

                for _, text, _ in topns[x]:
                    if answer in text:

                        answer = text
                        break
            results.append(names[x],answer)

        return results
    
    def reference(self,queries,answers,topns):

        terms = self.terms(queries)
        outputs =[]

        for x, (name,answer) in answers:
            
            topn,reference = topns[x], None

            if topn:

                query = f"{terms[x]} {answers[x][1]}"

                scores,_,_ = self.score([query],[text for _,text,_ in topn])

                index = scores[0][0][0]

                reference = topn[index[0]]

            outputs.append((name,answer,topn))
        return outputs

    def terms(self,queries):

        return self.similarity.batchterms(queries) if hasattr(self.similarity,"batchsterms") else(queries)






         

    



    



        



        