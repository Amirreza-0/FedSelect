# FedSelect
Unde rconstruction...

Federated feature selection and dimentionality reduction based on correlation
A distributed approach for performing feature selection across multiple datasets while maintaining data privacy. This implementation enables correlation-based feature selection where each site independently processes its local data while contributing to a global feature importance mapping.

## Overview

The system operates in a federated manner, with each site maintaining and processing its own data locally. The process is divided into three main components: local correlation analysis at each site, global mapping generation at a central coordinator, and feature selection based on the global mapping.

Consider a healthcare scenario where multiple hospitals want to collaboratively build a feature selection model without sharing patient data. Each hospital maintains its dataset locally but contributes to a shared understanding of feature importance.

## Process Flow

### Local Analysis
Each site independently analyzes its data to identify correlated features. For example, in a patient dataset, blood pressure and heart rate might be correlated. The site produces output like:

```python
local_correlations = {
    "group1": ["blood_pressure", "heart_rate", "pulse"],
    "group2": ["height", "weight", "bmi"],
    "group3": ["glucose", "insulin", "diabetes_risk"]
}
```

### Global Ranking
The central coordinator receives correlation groups from all sites and generates importance rankings. For instance:

```python
# Site 1 Data: Patient vitals
vitals_correlations = {
    "group1": ["systolic_bp", "diastolic_bp", "mean_bp"],
    "group2": ["temperature", "fever_risk"]
}

# Site 2 Data: Lab results
labs_correlations = {
    "group1": ["glucose", "insulin", "hba1c"],
    "group2": ["systolic_bp", "pulse"]
}

# Generated Global Raning
global_feature_ranks = {
    "systolic_bp": 1,    # Appears in both sites, good representative
    "diastolic_bp": 2,   # Highly correlated with systolic
    "glucose": 1,        # Main representative for lab group
    "temperature": 1,    # Unique in its group
    "pulse": 2,          # Less frequent
    "insulin": 2,        # Correlated with glucose
    "hba1c": 3,         # Less frequent
    "fever_risk": 2,    # Correlated with temperature
    "mean_bp": 3        # Derived from other BP measures
}
```

Lower ranks indicate features that are more consistently grouped and representative.

### Feature Selection
When a site needs to select features, it:
1. Performs local correlation analysis
2. Uses the global ranking to select the most representative features

For example:
```python
# Local correlations at Hospital A
vitals_correlations = {
    "group1": ["systolic_bp", "diastolic_bp", "mean_bp"],
    "group2": ["temperature", "fever_risk"]
}

# Using global ranking for feature slection
selected_features = ["systolic_bp", "temperature"]  # Lowest ranks from each group
```


## Privacy and Security

The system maintains privacy by design. Sites never share raw data - only feature correlation groups are transmitted. The global mapping contains no sensitive information, only feature names and their importance ranks. Each site can validate the mapping before applying it to their data.

## License

This project is licensed under the MIT License. See LICENSE.md for details.
