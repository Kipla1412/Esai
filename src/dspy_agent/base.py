import dspy
from abc import ABC, abstractmethod
import mlflow

from mlflow.entities import SpanType

class BaseMedicalModule(ABC):

    """ Base class used for medical related modules with Mlflow tracing , autolog"""
    def __init__(self, signature, module_name: str ="base"):

        self.signature =signature
        self.chain = dspy.ChainOfThought(signature)
        self.module_name = module_name

        mlflow.dspy.autolog()

    @abstractmethod

    def forward(self, **kwargs):
        """This method is to be implemented bu subclasses"""
        pass

    @mlflow.trace(name="module_execution", span_type =SpanType.TOOL)
    def __call__(self, **kwargs):
        
        """Make module callable with MLflow tracing and autolog"""
        with mlflow.start_span(f"{self.module_name}_execution", span_type=SpanType.TOOL):
            mlflow.set_tag("module", self.module_name)
            mlflow.set_tag("use_case", "medical")
            
            try:
                result = self.forward(**kwargs)
                
                mlflow.log_metric(f"{self.module_name}_output_length", len(str(result)))
                mlflow.set_tag(f"{self.module_name}_success", True)
                
                return result
            
            except Exception as e:
                mlflow.log_param(f"{self.module_name}_error", str(e))
                mlflow.set_tag(f"{self.module_name}_success", False)
                raise
