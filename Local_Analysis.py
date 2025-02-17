import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import defaultdict


@dataclass
class CorrelationGroup:
    features: List[str]
    mean_correlation: float
    is_isolated: bool = False


class UnionFind:
    """A simple union-find data structure for grouping elements."""
    def __init__(self, elements: List[str]):
        self.parent = {elem: elem for elem in elements}

    def find(self, item: str) -> str:
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, a: str, b: str) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self.parent[root_b] = root_a


class LocalCorrelationAnalyzer:
    def __init__(self, correlation_threshold: float = 0.8):
        """
        Initialize the analyzer with a minimum absolute correlation threshold.
        """
        self.correlation_threshold = correlation_threshold

    def calculate_correlation_matrix(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the absolute correlation matrix.
        """
        return data.corr().abs()

    def find_correlated_pairs(self, corr_matrix: pd.DataFrame) -> List[Tuple[str, str, float]]:
        """
        Identify all feature pairs with correlation above the threshold.
        Only the upper triangle of the matrix is considered to avoid duplicates.
        """
        pairs = []
        # Use the upper triangle to avoid redundant pairs
        for i, row in enumerate(corr_matrix.index):
            for j, col in enumerate(corr_matrix.columns):
                if j <= i:
                    continue
                corr_value = corr_matrix.iat[i, j]
                if corr_value >= self.correlation_threshold:
                    pairs.append((row, col, corr_value))
        return pairs

    def group_correlated_features(self, pairs: List[Tuple[str, str, float]], features: List[str]
                                 ) -> Dict[int, CorrelationGroup]:
        """
        Group features based on correlated pairs using union-find.
        Then, compute the mean correlation per group and mark isolated features.
        """
        uf = UnionFind(features)
        for f1, f2, _ in pairs:
            uf.union(f1, f2)

        # Group features by their root
        groups = defaultdict(list)
        for feat in features:
            groups[uf.find(feat)].append(feat)

        # Compute the mean correlation for pairs within each group
        group_results = {}
        for group in groups.values():
            # Select pairs where both features belong to the group
            group_corrs = [corr for f1, f2, corr in pairs if f1 in group and f2 in group]
            mean_corr = np.mean(group_corrs) if group_corrs else 1.0
            is_isolated = len(group) == 1 and not group_corrs
            group_results[len(group_results)] = CorrelationGroup(
                features=sorted(group),
                mean_correlation=mean_corr,
                is_isolated=is_isolated
            )
        return group_results

    def analyze(self, data: pd.DataFrame) -> Dict[int, CorrelationGroup]:
        """
        Execute the full analysis: compute correlation, find pairs, and group features.
        """
        corr_matrix = self.calculate_correlation_matrix(data)
        pairs = self.find_correlated_pairs(corr_matrix)
        features = list(data.columns)
        return self.group_correlated_features(pairs, features)


def format_local_results(correlation_groups: Dict[int, CorrelationGroup]) -> Dict[int, List[str]]:
    """
    Format the analysis output to show only the list of features per group.
    """
    return {gid: group.features for gid, group in correlation_groups.items()}


# --- Example usage ---
if __name__ == '__main__':
    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [1.1, 2.1, 3.1, 4.1, 5.1],  # Highly correlated with feature1
        'feature3': [10, 20, 30, 40, 50],
        'feature4': [11, 21, 31, 41, 51],         # Highly correlated with feature3
        'feature5': [100, 4, 300, 2, 500]          # Isolated feature
    })

    analyzer = LocalCorrelationAnalyzer(correlation_threshold=0.95)
    correlation_groups = analyzer.analyze(data)
    results = format_local_results(correlation_groups)
    print(results)
    
    # Output example:
    # {{0: ['feature4', 'feature2', 'feature1', 'feature3'], 1: ['feature5']}}
