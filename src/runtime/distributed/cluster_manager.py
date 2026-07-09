class ClusterManager:
    """
    Distributed AIOS Manager.
    Handles elastic scaling, fault recovery, and job dispatch across the AIOT Global Production Framework.
    """
    def __init__(self):
        self.active_nodes = []
        self.node_health_status = {}

    def register_node(self, node_id: str, hardware_capabilities: list):
        """Registers a new compute node (Edge device, GPU cluster, QPU)."""
        self.active_nodes.append(node_id)
        self.node_health_status[node_id] = "HEALTHY"
        print(f"[Distributed Manager] Node Registered: {node_id} (Hardware: {hardware_capabilities})")

    def schedule_execution_graph(self, graph_id: str):
        """
        Dispatches the Universal AI IR execution graph to the most optimal node.
        """
        if not self.active_nodes:
            raise RuntimeError("[Distributed Manager] FATAL: No active compute nodes available.")
            
        # Simplistic load balancing simulation
        target_node = self.active_nodes[0]
        print(f"[Distributed Manager] Dispatching Execution Graph '{graph_id}' to Node: {target_node}")
        return target_node

    def handle_node_failure(self, node_id: str):
        """
        Simulates Chaos Engineering and Auto-Recovery (Reliability Engineering).
        """
        print(f"\n[Distributed Manager] WARNING: Heartbeat lost for Node {node_id}")
        self.node_health_status[node_id] = "DEAD"
        self.active_nodes.remove(node_id)
        
        print(f"[Distributed Manager] Initiating Auto-Recovery... Transferring state to backup node.")
