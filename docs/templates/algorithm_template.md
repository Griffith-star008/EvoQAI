# Unified Algorithm Template

*Use this template to document all major algorithms within the AQIP framework.*

---

## 1. Background
*Contextualize the algorithm. Why is this specific procedure necessary within the larger framework?*

## 2. Problem Statement
*Define the exact mathematical or computational problem the algorithm solves.*

## 3. Inputs
*Define all arguments, their types, and dimensionalities.*
- $x$: Description

## 4. Outputs
*Define the return values and post-conditions.*
- $y$: Description

## 5. Assumptions
*List pre-conditions that must hold true for the algorithm to function correctly.*
- e.g., The input graph must be a DAG (no cycles).

## 6. Pseudo-code
*Provide clear, language-agnostic pseudo-code.*
```text
FUNCTION Name(inputs):
    // logic
    RETURN outputs
```

## 7. Complexity Analysis
*Analyze asymptotic bounds.*
- **Time Complexity:** $\mathcal{O}(\dots)$
- **Space Complexity:** $\mathcal{O}(\dots)$
- **Communication Complexity:** (If distributed)

## 8. Correctness
*Provide a sketch or formal proof that the algorithm terminates and produces the correct output.*

## 9. Experimental Results
*Summarize how this algorithm performs empirically.*

## 10. Failure Cases
*Describe specific conditions where the algorithm fails or degrades.*

## 11. Limitations
*Known boundaries of the algorithm's applicability.*

## 12. Future Work
*Proposed improvements or relaxations of current assumptions.*
