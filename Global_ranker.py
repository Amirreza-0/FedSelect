from collections import defaultdict
from typing import Dict, List

class GlobalRanksGenerator:
    def __init__(self):
        """
        Initialize the global ranks generator.
        Tracks feature occurrences across all sites.
        """
        self.feature_counts = defaultdict(int)
        self.feature_groups = defaultdict(set)
        
    def update_from_local_results(self, local_results: Dict[int, List[str]], site_id: str) -> None:
        """
        Update feature statistics from local correlation results.
        
        Args:
            local_results: Dictionary of group_id to feature lists
            site_id: Unique identifier for the site
        """
        for group_id, features in local_results.items():
            group_key = f"{site_id}_{group_id}"
            for feature in features:
                self.feature_counts[feature] += 1
                self.feature_groups[feature].add(group_key)
    
    def calculate_global_ranks(self) -> Dict[str, int]:
        """
        Calculate global ranks for features.
        Ranks start from 1 (most important) and increase.
        
        Returns:
            Dictionary mapping features to their ranks
        """
        # Calculate initial scores (higher score = more frequent/important)
        feature_scores = {
            feature: len(groups) + count
            for feature, (count, groups) in 
            ((f, (self.feature_counts[f], self.feature_groups[f])) 
             for f in self.feature_counts)
        }
        
        # Convert scores to ranks (higher score = lower rank)
        sorted_features = sorted(
            feature_scores.items(),
            key=lambda x: (-x[1], x[0])  # Sort by score desc, then feature name
        )
        
        # Assign ranks
        initial_ranks = {
            feature: rank 
            for rank, (feature, _) in enumerate(sorted_features, 1)
        }
        
        # Ensure ranks start from 1
        min_rank = min(initial_ranks.values())
        normalized_ranks = {
            feature: (rank - min_rank) + 1
            for feature, rank in initial_ranks.items()
        }
        
        return normalized_ranks

# Example usage:
if __name__ == "__main__":
    # Initialize generator
    generator = GlobalRanksGenerator()
    
    # Example local results from two sites
    site1_results = {
        0: ["feature1", "feature2", "feature4"],
        1: ["feature3"],
        2: ["feature5"]
    }
    
    site2_results = {
        0: ["feature1", "feature2"],
        1: ["feature3", "feature5", "feature4"]
    }
    
    # Update with results from multiple sites
    generator.update_from_local_results(site1_results, "site1")
    generator.update_from_local_results(site2_results, "site2")
    
    # Get global ranks
    ranks = generator.calculate_global_ranks()
    print(ranks)
    # Output example:
    # {
    #     'feature1': 1,
    #     'feature2': 1,
    #     'feature4': 2,
    #     'feature3': 3,
    #     'feature5': 3
    # }
