import yaml
import logging
from src.agent.base import Agent
import mlflow
from mlflow.entities import SpanType
import asyncio
import dspy
import json
from datasets import load_dataset
import os
from dotenv import load_dotenv

load_dotenv()

#mlflow.set_experiment("convo_session")
class MedicalQASignature(dspy.Signature):
    """
    Given a user's medical question, produce a structured JSON response.

    Output format (strictly follow this):
    {
        "answer": "<clear, safe medical response>",
        "confidence": "<float between 0 and 1>",
        "category": "<type of question, e.g. symptom_advice, general_info, mental_health>"
    }
    """
    question = dspy.InputField(desc="User's medical question")
    answer = dspy.OutputField(desc="JSON output with answer, confidence, and category")

class ConversationalAgent:
    def __init__(self, config_path: str = r"D:\backend\ESAI\src\dspy_agent\dspy.yml"):
        
        self.logger = logging.getLogger("ConversationalAgent")
        logging.basicConfig(level=logging.INFO)
        self.logger.setLevel(logging.DEBUG)

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.llm_config = config.get("llm", {})
        self.agent_config = config.get("agent", {})

        lm = dspy.LM(model ="gemini/gemini-2.5-flash", api_key= os.getenv("GEMINI_API_KEY"))

        dspy.configure(lm=lm)

        mlflow.dspy.autolog(log_compiles=True, log_evals=True, log_traces_from_compile=True)

        dataset = load_dataset("NeuML/pubmed-h5n1", split="train[:1000]")
        self.train_examples = self.build_training(dataset)

        self.medical_predictor = dspy.Predict(MedicalQASignature)
        self._init_optimizer()

        full_config = {
            **self.agent_config,
            "model": self.llm_config,
            "tools": []
        }
        self.agent = Agent(**full_config)
        
        self.conversation_history = ""
        self.initial_message = "Hello! I am your medical assistant. How can I help you today?"

        self.logger.info("Conversational Agent initialized successfully")

    def build_training(self, dataset):

        examples = []
        for item in dataset:
            question_text = item.get("title", "") or item.get("abstract", "")
            answer_dict = {
                "answer": f"This is a medical summary about: {question_text[:100]}.",
                "confidence": 0.9,
                "category": "general_info" 
            }
            example = dspy.Example(
                question=question_text,
                answer=json.dumps(answer_dict)
            ).with_inputs("question")
            examples.append(example)
        return examples

    def evaluation_metric(self, example, pred, trace=None):
        try:
            gold = json.loads(example.answer)
            pred = json.loads(pred.answer)
            return int(gold.get("category") == pred.get("category"))
        except Exception:
            return 0

    def _init_optimizer(self):
        try:
            optimizer = dspy.MIPROv2(metric=self.evaluation_metric, auto="light")
            self.medical_predictor = optimizer.compile(self.medical_predictor, trainset=self.train_examples)
            self.logger.info("DSPy optimizer compiled successfully with PubMed training data")
        except Exception as e:
            self.logger.error(f"Optimizer initialization failed: {e}")
    #
    def reset_conversation(self):
        """Clear the conversation history"""
        self.logger.info("Resetting conversation history")
        self.conversation_history = ""

   
    @mlflow.trace(name="generate_response", span_type=SpanType.AGENT)
    async def generate_response(self, user_input: str) -> str:
        """
        Generates a conversational response using DSPy for reasoning
        and the Agent for contextual reply.
        """
        try:
            
            self.conversation_history += f"User: {user_input}\n"

            dspy_result = await asyncio.to_thread(
                self.medical_predictor, question=user_input
            )
            # dspy_answer = dspy_result.answer.strip()
            raw_answer = dspy_result.answer.strip()

            try:
                parsed = json.loads(raw_answer)
                dspy_answer = parsed.get("answer", raw_answer)
                mlflow.set_tag("json_confidence", parsed.get("confidence", 0.0))
                mlflow.set_tag("json_category", parsed.get("category", "unknown"))
            except json.JSONDecodeError:
                dspy_answer = raw_answer

            base_response = await asyncio.to_thread(
                self.agent, "conversation_agent", text=self.conversation_history
            )

          
            final_response = f"{dspy_answer}\n\n(Contextual reply: {base_response})"
            self.conversation_history += f"Agent: {final_response}\n"

           
            mlflow.set_tag("user_input", user_input)
            mlflow.set_tag("dspy_reasoning", dspy_answer)
            mlflow.set_tag("agent_output", base_response)
            mlflow.log_metric("conversation_length", len(self.conversation_history))

            self.logger.debug(f"Final response generated: {final_response}")
            return final_response

        except Exception as e:
            self.logger.error(f"Error during agent response generation: {e}")
            mlflow.log_param("error", str(e))
            return "Sorry, I encountered an error while processing your request."

 
    def get_initial_message(self):
        return self.initial_message

    @staticmethod
    def log_static_params(agent):
        """Log static configuration details into MLflow"""
        mlflow.log_param("llm_method", agent.llm_config.get("method"))
        mlflow.log_param("llm_model", agent.llm_config.get("path"))
        mlflow.log_param("llm_temperature", agent.llm_config.get("temperature"))
        mlflow.log_param("agent_description", agent.agent_config.get("description"))

        max_iter = agent.agent_config.get("max_iterations")
        if max_iter is not None:
            mlflow.log_param("max_iterations", max_iter)

def test_agent_with_real_data(agent_instance):
    # Use first example from your already loaded training dataset
    example = agent_instance.train_examples[0]

    # Generate prediction for example input question using the current predictor
    pred = agent_instance.medical_predictor(question=example.question)

    # Evaluate metric score on this example-prediction pair
    score = agent_instance.evaluation_metric(example, pred)
    print(f"Evaluation metric score: {score}")

    # Re-run optimizer compilation on real training data to ensure no failures
    agent_instance._init_optimizer()
    print("Optimizer compilation succeeded.")

    # Generate an agent response for the example question asynchronously
    response = asyncio.run(agent_instance.generate_response(example.question))
    print(f"Generated response:\n{response}")

if __name__ == "__main__":
    # Initialize your agent as usual
    agent = ConversationalAgent()

    # Run the test using real loaded training data inside your agent
    test_agent_with_real_data(agent)
# if __name__ == "__main__":
   
#     if mlflow.active_run() is not None:
#         mlflow.end_run()

#     agent_instance = ConversationalAgent()
#     print("Medical Assistant is ready. Type your question below.\n")

#     with mlflow.start_run(run_name="convo_session"):
#         agent_instance.log_static_params(agent_instance)

#         while True:
#             try:
#                 user_input = input("You: ").strip()
#                 if not user_input:
#                     continue

#                 if user_input.lower() in {"exit", "quit"}:
#                     print("\n Agent: Thank you for chatting! Take care and stay healthy ")
#                     break

#                 response = asyncio.run(agent_instance.generate_response(user_input))

#                 print(f"Agent: {response}\n")

#             except KeyboardInterrupt:
#                 print("\nSession ended by user.")
#                 break

#             except Exception as e:
#                 print("Error during agent processing:", e)
#                 mlflow.log_param("runtime_error", str(e))
#                 break
