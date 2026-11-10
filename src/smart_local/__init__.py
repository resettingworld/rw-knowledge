from smart_local.config import settings
from smart_local.sdk.client import SmartLocalSDK
from smart_local.manager.registry import KnowledgeManager

__version__ = settings.version
__all__ = ["settings", "SmartLocalSDK", "KnowledgeManager"]
