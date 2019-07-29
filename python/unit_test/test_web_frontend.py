from rlgames.agents.random_fast import FastRandomAgent
from rlgames.httpfrontend.server import get_web_app

random_agent = FastRandomAgent(9)
web_app = get_web_app({'random' : random_agent})
web_app.run()
