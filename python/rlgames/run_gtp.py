import h5py

from rlgames.gtp import GTPFrontend
from rlgames.agent.predict import load_prediction_agent
from rlgames.agent import termination

model_file = h5py.File('/home/armandli/rlgames/data/checkpoints/sevenplane_large_agent.h5') #TODO
agent = load_prediction_agent(model_file)
strategy = termination.get('opponent_passes')
termination_agent = termination.TerminationAgent(agent, strategy)

frontend = GTPFrontend(termination_agent)
frontend.run()
