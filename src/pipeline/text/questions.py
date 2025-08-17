from ..hfpipeline import HFPipeline

class Questions(HFPipeline):

    def __init__(self,path=None,quantize = False,gpu =True,model=None,**kwargs):

        super().__init__("question-answering",path,quantize,gpu,model,**kwargs)

    def  __call__(self,questions,contexts, workers =0):

        answers =[]

        for x, question in enumerate(questions):
            if question and contexts[x]:
                result = self.pipeline(question =question, context=contexts[x],num_workers = workers)

                answer, score = result['answer'], result['score']

                if score < 0.05:
                    answer =None
                
                answers.append(answer)

            else:

                answers.append(None)

        return answers

    