from typing import Dict, List


class FeatureSelector:
    def __init__(self):
        """
        Initialize the feature selector.
        """
        pass

    def select_features(
            self,
            local_groups: Dict[int, List[str]],
            global_ranks: Dict[str, int]
    ) -> List[str]:
        """
        Select features based on local correlation groups and global ranks.
        Takes at least one feature from each local group, selecting the one with lowest rank.

        Args:
            local_groups: Dictionary of local correlation groups
            global_ranks: Dictionary of global feature ranks

        Returns:
            List of selected feature names
        """
        selected_features = set()

        # Process each local group
        for group_features in local_groups.values():
            if not group_features:
                continue

            # Find feature with lowest rank in this group
            best_feature = min(
                group_features,
                key=lambda f: global_ranks.get(f, float('inf'))
            )

            selected_features.add(best_feature)

        return sorted(list(selected_features))


# Example usage:
if __name__ == "__main__":
    # Complete pipeline example
    def feature_selection_pipeline(data, site_id):
        # 1. Local correlation analysis
        # local_analyzer = LocalCorrelationAnalyzer(correlation_threshold=0.8)
        # local_groups = local_analyzer.analyze(data)

        # 2. Update global ranks
        # ranks_generator = GlobalRanksGenerator()
        # ranks_generator.update_from_local_results(local_groups, site_id)
        # global_ranks = ranks_generator.calculate_global_ranks()


        # 3. Select features
        selector = FeatureSelector()
        selected_features = selector.select_features(local_groups, global_ranks)

        return selected_features

    # Example usage

    # example dataset
    dataset= {
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [5, 4, 3, 2, 1],
        "feature3": [1, 1, 1, 1, 1],
        "feature4": [2, 2, 2, 2, 2],
        "feature5": [3, 3, 3, 3, 3],
        "feature6": [4, 4, 4, 4, 4],
        "feature7": [5, 5, 5, 5, 5],
        "feature8": [6, 6, 6, 6, 6]
    }


    # Local correlation groups generated from local analysis
    local_groups = {
                0: ["feature1", "feature2", "feature4"],
                1: ["feature3", "feature5"],
                2: ["feature6", "feature7", "feature8"]
            }

    # generated from previous runs using global ranking
    global_ranks = {
                "feature1": 3,
                "feature2": 2,
                "feature3": 1,
                "feature4": 4,
                "feature5": 5,
                "feature6": 6,
                "feature7": 7,
                "feature8": 8
            }

    # Feature selection
    selector = FeatureSelector()
    selected_features = selector.select_features(local_groups, global_ranks)

    print("dataset features:", dataset.keys())
    print("Selected features:", selected_features)
    # Output example: ['feature2', 'feature3', 'feature6']

    # compare dataset and selected_features, compare the length of the selected features and the dataset as percentage
    percentage = (1- (len(selected_features) / len(dataset.keys()))) * 100
    print(f"Percentage of feature reduction: {percentage:.2f}%")

