"""
Base Agent class
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Base class for all agents"""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    def process(self, **kwargs) -> Dict[str, Any]:
        """
        Process input and return results
        
        Returns:
            Dict containing processing results and metadata
        """
        pass
    
    def log(self, message: str, level: str = "INFO"):
        """Log a message"""
        getattr(self.logger, level.lower())(message)

