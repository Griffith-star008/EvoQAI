class HierarchicalMemory:
    """
    Layer 6: Hierarchical Memory
    Simulates AGI-like memory hierarchy: Working (Execution), Semantic (Concepts), 
    Procedural (Policies), Long-term (Experience Graph).
    """
    def __init__(self):
        self.working_memory = {}
        self.semantic_memory = {"overheating": "high risk of hardware fault"}
        self.procedural_memory = {"compression": "reduce gates"}
        self.long_term_memory = []

    def consolidate_memory(self, execution_context: dict, outcome: float):
        """
        Moves transient working memory into long-term storage if the outcome is significant.
        """
        self.working_memory = execution_context
        
        print("[Hierarchical Memory] Consolidating working memory into long-term storage...")
        if outcome > 0.8:
            self.long_term_memory.append({"context": execution_context, "outcome": outcome})
            print(f"[Hierarchical Memory] Experience saved. Total Long-Term Memories: {len(self.long_term_memory)}")
