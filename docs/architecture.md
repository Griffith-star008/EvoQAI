# AQIP Architecture

This document provides visual representations of the AQIP framework architecture using Mermaid.js diagrams.

## 1. Universal AI Intermediate Representation (UAIR)

The UAIR compiler unifies classical tensors, quantum gates, and agent decisions into a single SSA DAG.

```mermaid
graph TD
    subgraph Frontend [Language Frontends]
        PT[PyTorch / JAX]
        QK[Qiskit / Cirq]
        AG[Agent SDK]
    end

    subgraph Compiler [UAIR Compiler]
        Parser[Parser & AST]
        SSA[SSA Graph Constructor]
        TypeCheck[Cross-Domain Type Checker]
    end

    subgraph IR [Unified Representation]
        DAG((UAIR DAG))
    end

    PT --> Parser
    QK --> Parser
    AG --> Parser
    Parser --> SSA
    SSA --> TypeCheck
    TypeCheck --> DAG
```

## 2. Self-Evolving Runtime Engine

The core innovation of AQIP is the autonomous, mathematically verified runtime optimization loop.

```mermaid
stateDiagram-v2
    [*] --> Monitor
    Monitor --> Diagnose : Telemetry Data
    Diagnose --> Propose : Hot Subgraph
    
    state Propose {
        direction LR
        PolicyNet --> CandidateList
    }
    
    Propose --> Verify : Permutation π
    
    state Verify {
        direction LR
        EncodeSMT --> CheckEquivalence
        CheckEquivalence --> SAFE
        CheckEquivalence --> UNSAFE
    }
    
    Verify --> Deploy : SAFE (UNSAT)
    Verify --> Propose : UNSAFE (SAT) / TIMEOUT
    
    Deploy --> Simulate : Update Graph
    Simulate --> Monitor : Execution
```

## 3. Distributed Cluster Architecture

How AQIP maps the UAIR graph onto heterogeneous hardware clusters via Kubernetes.

```mermaid
graph LR
    subgraph Master [Control Plane]
        Sched[Global Scheduler]
        Twin[Digital Twin Simulator]
    end

    subgraph Worker1 [GPU Node]
        W1[Local Executor]
        A100[NVIDIA A100]
    end

    subgraph Worker2 [QPU Node]
        W2[Quantum Controller]
        Eagle[IBM Eagle QPU]
    end

    Sched -->|Sub-graph A| W1
    Sched -->|Sub-graph B| W2
    W1 --> A100
    W2 --> Eagle
    Twin -.->|Feedback| Sched
```
