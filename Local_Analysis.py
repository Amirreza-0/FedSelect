import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass


@dataclass
class CorrelationGroup:
    features: List[str]
    mean_correlation: float
    is_isolated: bool = False


class LocalCorrelationAnalyzer:
    def __init__(self, correlation_threshold: float = 0.8):
        """
        Initialize the local correlation analyzer.

        Args:
            correlation_threshold: Minimum absolute correlation to consider features as correlated
        """
        self.correlation_threshold = correlation_threshold

    def calculate_correlation_matrix(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the correlation matrix for the input dataset.

        Args:
            data: Input DataFrame containing features

        Returns:
            Correlation matrix as DataFrame
        """
        return data.corr().abs()

    def find_correlated_pairs(self, correlation_matrix: pd.DataFrame) -> List[Tuple[str, str, float]]:
        """
        Find pairs of features that are highly correlated.

        Args:
            correlation_matrix: Correlation matrix DataFrame

        Returns:
            List of tuples containing (feature1, feature2, correlation_value)
        """
        pairs = []
        # Get upper triangle of correlation matrix
        upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

        # Find feature pairs with correlation above threshold
        for col in upper_tri.columns:
            for idx, value in upper_tri[col].items():
                if value >= self.correlation_threshold:
                    pairs.append((idx, col, value))

        return pairs

    def find_isolated_features(self, data: pd.DataFrame, grouped_features: Set[str]) -> List[str]:
        """
        Find features that are not part of any correlation group.

        Args:
            data: Input DataFrame
            grouped_features: Set of features already in correlation groups

        Returns:
            List of isolated feature names
        """
        all_features = set(data.columns)
        return list(all_features - grouped_features)

    def group_correlated_features(self,
                                  pairs: List[Tuple[str, str, float]],
                                  data: pd.DataFrame) -> Dict[int, CorrelationGroup]:
        """
        Group correlated features together and identify isolated features.

        Args:
            pairs: List of correlated feature pairs
            data: Input DataFrame for finding isolated features

        Returns:
            Dictionary mapping group IDs to CorrelationGroup objects
        """
        # Initialize groups
        groups: Dict[int, Set[str]] = {}
        correlations: Dict[int, List[float]] = {}
        group_id = 0

        # Helper function to find group containing a feature
        def find_feature_group(feature: str) -> int:
            for gid, group in groups.items():
                if feature in group:
                    return gid
            return -1

        # Process each correlation pair
        for feat1, feat2, corr in pairs:
            group1 = find_feature_group(feat1)
            group2 = find_feature_group(feat2)

            if group1 == -1 and group2 == -1:
                # Create new group
                groups[group_id] = {feat1, feat2}
                correlations[group_id] = [corr]
                group_id += 1
            elif group1 == -1:
                # Add to existing group2
                groups[group2].add(feat1)
                correlations[group2].append(corr)
            elif group2 == -1:
                # Add to existing group1
                groups[group1].add(feat2)
                correlations[group1].append(corr)
            elif group1 != group2:
                # Merge groups
                groups[group1].update(groups[group2])
                correlations[group1].extend(correlations[group2])
                del groups[group2]
                del correlations[group2]

        # Get all grouped features
        grouped_features = set()
        for feature_group in groups.values():
            grouped_features.update(feature_group)

        # Find isolated features
        isolated_features = self.find_isolated_features(data, grouped_features)

        # Create result dictionary with both grouped and isolated features
        result = {}

        # Add correlated groups
        for gid, features in groups.items():
            mean_corr = np.mean(correlations[gid])
            result[gid] = CorrelationGroup(
                features=list(features),
                mean_correlation=mean_corr,
                is_isolated=False
            )

        # Add isolated features as individual groups
        for i, feature in enumerate(isolated_features, start=len(groups)):
            result[i] = CorrelationGroup(
                features=[feature],
                mean_correlation=1.0,  # Self-correlation
                is_isolated=True
            )

        return result

    def analyze(self, data: pd.DataFrame) -> Dict[int, CorrelationGroup]:
        """
        Perform complete correlation analysis on the dataset.

        Args:
            data: Input DataFrame

        Returns:
            Dictionary of correlation groups including isolated features
        """
        correlation_matrix = self.calculate_correlation_matrix(data)
        correlated_pairs = self.find_correlated_pairs(correlation_matrix)
        correlation_groups = self.group_correlated_features(correlated_pairs, data)
        return correlation_groups


def format_local_results(correlation_groups: Dict[int, CorrelationGroup]) -> dict[int, list[str]]:
    """
    Format correlation groups into a detailed output format.

    Args:
        correlation_groups: Dictionary of CorrelationGroup objects

    Returns:
        Dictionary with correlation groups and metadata
    """
    # return {
    #     "groups": {
    #         gid: {
    #             "features": group.features,
    #             "mean_correlation": group.mean_correlation,
    #             "is_isolated": group.is_isolated
    #         }
    #         for gid, group in correlation_groups.items()
    #     }
    # }

    return {
            gid: group.features
            for gid, group in correlation_groups.items()
        }


# Example usage
data = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5],
    'feature2': [1.1, 2.1, 3.1, 4.1, 5.1],  # Correlated with feature1
    'feature3': [10, 20, 30, 40, 50],
    'feature4': [11, 21, 31, 41, 51],  # Correlated with feature3
    'feature5': [100, 4, 300, 2, 500]  # Isolated feature
})

analyzer = LocalCorrelationAnalyzer(correlation_threshold=0.95)
correlation_groups = analyzer.analyze(data)
results = format_local_results(correlation_groups)

print(results)
# Output example:
# {{0: ['feature4', 'feature2', 'feature1', 'feature3'], 1: ['feature5']}}
