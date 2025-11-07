import dspy

class MedicalConversationSignature(dspy.Signature):

    """Signature for conversational interactions"""
    user_input: str = dspy.InputField()
    conversation_context: str = dspy.InputField()
    response: str = dspy.OutputField()

class MedicalQA(dspy.Signature):

    """Signature for medical question and answering"""

    question: str = dspy.InputField()
    context: str = dspy.InputField(default="")
    answer: str = dspy.OutputField()


class SymptomsAnalysisSignature(dspy.Signature):
    """Analyze medical symptoms"""
    symptoms: str = dspy.InputField()
    patient_history: str = dspy.InputField(default="")
    analysis: str = dspy.OutputField()


class Summarizer(dspy.Signature):

    """Signature for text summarization"""
    text: str = dspy.InputField()
    summary: str = dspy.OutputField()

class ValidatorSignature(dspy.Signature):

    """Signature for input validation"""
    input_text: str = dspy.InputField()
    is_valid: bool = dspy.OutputField()
    validation_message: str = dspy.OutputField()

