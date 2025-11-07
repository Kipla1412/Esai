# from trulens_eval import Tru, TruCustomApp

# def my_agent(prompt: str):

#     return f"user said: {prompt}"

# tru = Tru(database_url="sqlite:///default.sqlite")
# tru_app =TruCustomApp(my_agent, app_id ="first_agent")
# with tru_app.run("check_name") as rec:
#     output = my_agent("Hello Umar!")
#     rec.inputs = {"prompt": "Hello Umar!"}
#     rec.outputs = {"response": output}
# # Launch dashboard
# tru.run_dashboard()
# from trulens.core import Tru
# from trulens.apps.app import TruApp
# from trulens.dashboard import run_dashboard
# # 1. Define your function
# def my_agent(prompt: str):
#     return f"user said: {prompt}"

# # 2. Initialize Tru + new App
# tru = Tru(database_url="sqlite:///default.sqlite")
# tru_app = TruApp(app=my_agent, app_id="second_agent")

# # 3. Record a run
# with tru_app.run(run_name="check_name") as rec:
#     output = my_agent("Hello Umar!")
#     rec.inputs = {"prompt": "Hello Umar!"}
#     rec.outputs = {"response": output}

# # 4. Launch dashboard
# run_dashboard(port=8000)
from trulens.core import Tru
from trulens.apps.app import TruApp
from trulens.core import TruSession
# Your simple agent
def my_agent(prompt: str):
    return f"user said: {prompt}"

# Initialize Tru
tru = TruSession(database_url="sqlite:///default.sqlite")

# Create a TruApp
tru_app = TruApp(app=my_agent, app_id="my_agent_app", app_version="1.0.0",session=tru)

# Record a run
with tru_app.run(run_name="check_run") as rec:
    output = my_agent("Hello Umar!")
    rec.records.append({
        "inputs": {"prompt": "Hello Umar!"},
        "outputs": {"response": output}
    })


# Launch dashboard
tru.run_dashboard(port=8000)
