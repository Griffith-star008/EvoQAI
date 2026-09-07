import math

class ADWINDriftDetector:
    """
    Simplified Adaptive Windowing (ADWIN) Drift Detector.
    Tracks error rates and detects statistically significant concept drift.
    """
    def __init__(self, delta=0.1):
        self.delta = delta
        self.window = []
        self.total = 0.0

    def add_element(self, value: float) -> bool:
        """
        Adds a new element (e.g., error rate: 0.0 for correct, 1.0 for wrong).
        Returns True if concept drift is detected.
        """
        self.window.append(value)
        self.total += value
        
        # We only check for drift if window is reasonably large
        if len(self.window) < 10:
            return False
            
        return self._check_drift()

    def _check_drift(self) -> bool:
        n = len(self.window)
        
        # Partition window into two parts: W0 and W1
        # For performance, we don't check every possible split, just recent splits
        for i in range(1, n):
            n0 = i
            n1 = n - i
            
            total0 = sum(self.window[:i])
            total1 = self.total - total0
            
            u0 = total0 / n0
            u1 = total1 / n1
            
            m = 1.0 / (1.0/n0 + 1.0/n1)
            # Hoeffding bound epsilon
            epsilon = math.sqrt((1.0 / (2.0 * m)) * math.log(4.0 / self.delta))
            
            if abs(u0 - u1) > epsilon:
                # Drift detected! Cut window to W1
                self.window = self.window[i:]
                self.total = sum(self.window)
                return True
                
        return False

    def reset(self):
        self.window = []
        self.total = 0.0
