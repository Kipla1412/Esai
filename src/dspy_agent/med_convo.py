import dspy
import mlflow
from .base import BaseMedicalModule
from .signatures import MedicalConversationSignature

class MedicalConversationModule(BaseMedicalModule):
    """Mesical conversation module with mlflow and dspy autolog"""

    def __init__(self):

        super().__init__(MedicalConversationSignature, "medical_conversation")

        mlflow.dspy.autolog()

    def forward(self, user_input: str, conversation_context: str =""):
        """Generate medical conversation response - auto-logged by MLflow"""

        result =self.chain(
            user_input= user_input,
            conversation_context = conversation_context
        )

        return result.response
    
medical_conversation = MedicalConversationModule()