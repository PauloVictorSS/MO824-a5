from dataclasses import dataclass
from typing import List, Set

@dataclass
class ScQbfSolution:
    elements: List[int]
    value: float = None

    def __str__(self):
        return f"ScQbfSolution(value={self.value:.2f}, elements={len(self.elements)})"