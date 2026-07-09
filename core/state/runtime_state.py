import copy
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class FormalKnowledgeModel:
    semantic: dict = field(default_factory=dict)
    procedural: dict = field(default_factory=dict)
    episodic: dict = field(default_factory=dict)
    structural: dict = field(default_factory=dict)
    experimental: dict = field(default_factory=dict)
    theoretical: dict = field(default_factory=dict)
    consistency_score: float = 1.0

@dataclass
class RuntimeState:
    """
    Formal AQIP v6.0 Unified Runtime State Object.
    No module holds hidden internal states; everything passes through this object.
    """
    state_id: str = "init"
    identity: Dict[str, Any] = field(default_factory=dict)
    goal: Dict[str, float] = field(default_factory=dict)
    knowledge: FormalKnowledgeModel = field(default_factory=FormalKnowledgeModel)
    belief: Dict[str, float] = field(default_factory=dict)
    world_model: Dict[str, Any] = field(default_factory=dict)
    policy: str = "default_policy"
    reflection: Dict[str, Any] = field(default_factory=dict)
    resources: Dict[str, float] = field(default_factory=lambda: {"compute": 100.0, "memory": 100.0, "battery": 100.0})
    execution_history: list = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=lambda: {"utility": 0.0, "cost": 0.0, "accuracy": 0.0})
    
    def snapshot(self):
        """Returns a deep copy of the state for rollback purposes."""
        return copy.deepcopy(self)
