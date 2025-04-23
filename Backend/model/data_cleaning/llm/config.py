from langchain_groq.chat_models import ChatGroq
import os
from dotenv import load_dotenv
from pandasai import SmartDataframe as agent
from typing import Any, Dict, Optional

"""
This file creates a wrap around the SmartDataframe class from pandasai.
It help to integrate the LLM with the SmartDataframe class.
"""

load_dotenv()
try:
    chat_model = ChatGroq(model_name='llama3-70b-8192', api_key=os.environ['GROQ_API_KEY'])
except KeyError:
    raise EnvironmentError("Please set the 'GROQ_API_KEY' environment variable.")

def set_production():
    """
    """
    try:
        production_state = os.environ['PRODUCTION']
        if production_state.lower() == 'true':
            return True
        else:
            return False
    except KeyError:
        return False


def set_verbose():
        """
        Set the verbosity of the agent.
        :param verbose: Boolean value to set verbosity.
        """
        try:
            verbose_state = os.environ['VERBOSE']
            if verbose_state.lower() == 'true':
                return True
            else:
                return False
        except KeyError:
            return False

from pandasai.engine import set_pd_engine

set_pd_engine("pandas")


class Config:
    """
    Configuration class for the SmartDataframe.
    This class initializes the SmartDataframe with the provided data and configuration.
    """
    def __init__(self, data, config: Dict[str,Any]=None):
        self.data = data    
        self.config = config if config else {}
        self.config['llm'] = chat_model
        self.config['verbose'] = set_verbose()
        self.config['use_error_correction_framework'] = True
        self.config['conversational'] = True
        self.config['enable_cache'] = True
        self.config['cache_path'] = os.path.join(os.path.dirname(__file__), 'cache')
        self.agent = agent(data, config=self.config)

    def get_agent(self):
        return self.agent

    def get_data(self):
        return self.data
