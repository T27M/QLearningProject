import core.config as Config

# Used to initalise all enviroment values - see config.json
config = Config.Config()

# Instance of the QLearning agent
q_agent = QAgent.QAgent(config.get_config('QAgent'))
