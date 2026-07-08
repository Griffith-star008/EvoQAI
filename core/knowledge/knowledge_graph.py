import json
from typing import Dict, List, Any

class QuantumKnowledgeGraph:
    """
    Upgrade 2: Knowledge Engine - Knowledge Graph
    Builds a semantic graph linking Contexts (e.g., Overheating) -> Strategies -> QuantumIRs.
    """
    def __init__(self):
        # A simple adjacency list representation of the Knowledge Graph
        self.graph = {}

    def add_experience(self, context: str, strategy: str, circuit_depth: int, accuracy: float):
        """
        Extracts knowledge from a runtime experience and adds it to the graph.
        """
        if context not in self.graph:
            self.graph[context] = {"type": "CONTEXT", "edges": {}}
            
        if strategy not in self.graph[context]["edges"]:
            self.graph[context]["edges"][strategy] = {"avg_accuracy": 0.0, "avg_depth": 0.0, "count": 0}
            
        edge = self.graph[context]["edges"][strategy]
        
        # Incremental average update
        n = edge["count"]
        edge["avg_accuracy"] = (edge["avg_accuracy"] * n + accuracy) / (n + 1)
        edge["avg_depth"] = (edge["avg_depth"] * n + circuit_depth) / (n + 1)
        edge["count"] += 1
        
        print(f"[Knowledge Graph] Edge Updated: {context} -[{strategy}]-> Acc: {edge['avg_accuracy']:.2f}")

    def query_best_strategy(self, context: str) -> str:
        """
        Reasoning over the graph to find the best historically proven strategy for a context.
        """
        if context not in self.graph or not self.graph[context]["edges"]:
            return "STRATEGY_STANDARD" # Fallback
            
        best_strategy = max(
            self.graph[context]["edges"].items(),
            key=lambda x: x[1]["avg_accuracy"]
        )[0]
        
        print(f"[Knowledge Graph] Retrieval: For context '{context}', best strategy is '{best_strategy}'")
        return best_strategy

# Example usage
if __name__ == "__main__":
    kg = QuantumKnowledgeGraph()
    kg.add_experience("OVERHEATING_RISK", "STRATEGY_SHALLOW_CIRCUIT", circuit_depth=2, accuracy=0.88)
    kg.add_experience("OVERHEATING_RISK", "STRATEGY_DEEP_EVOLUTION", circuit_depth=10, accuracy=0.60) # Deep fails in overheating due to noise
    
    best = kg.query_best_strategy("OVERHEATING_RISK")
