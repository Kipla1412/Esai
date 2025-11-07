import mlflow
from mlflow.entities import SpanType

def init_mlflow_dspy_autolog(tracking_uri: str = "http://localhost:5000", experiment_name: str = "medical_assistant"):
    """Initialize MLflow with DSPy autolog enabled"""
    
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    mlflow.dspy.autolog()
    
    mlflow.autolog(silent=True)
    
    print(f" Mlflow initialized with DSPy autolog")
    print(f" Tracking URI: {tracking_uri}")
    print(f" Experiment: {experiment_name}")
