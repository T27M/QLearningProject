from core.config import Config
from core.enviroment import Environment
from core.featureprocessor import FeatureProcessor

from qlearning.qagent import QAgent


# Used to initalise all enviroment values - see config.json
config = Config()

# Instance of the QLearning agent
qagent = QAgent(config)

# Feature processor to extract feature vector
feature_processor = FeatureProcessor(config)

# The environment to run
env = Environment(config, feature_processor, qagent)

# Run the environment with the QAgent
env.run()
