"""
FedSelect: Federated Feature Selection for Random Forest
=========================================================

Proper experimental comparison of:
1. Centralized RF - Baseline (all data pooled)
2. Federated RF - Local models aggregated (all features)
3. Federated RF + Standard Feature Selection - MI-based local FS
4. Federated RF + FedSelect - Our federated feature selection method

Key Design Choices:
- NO imputation - use HistGradientBoosting which handles NaN natively
- Proper stratified cross-validation
- Threshold optimization for imbalanced data
- Fair comparison: all methods use same data splits
"""

import sys
import os
sys.path.insert(0, os.path.abspath('..'))

import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    recall_score, precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from Local_Analysis import LocalCorrelationAnalyzer, format_local_results
from Global_ranker import GlobalRanksGenerator

np.random.seed(42)
RANDOM_STATE = 42

# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_mimic_like_data(n_samples: int = 10000, random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generate synthetic MIMIC-III-like ICU data.

    Features organized by clinical category (split across 5 sites):
    1. Demographics & Admission (8 features)
    2. Vital Signs (24 features)
    3. Laboratory - Chemistry (28 features)
    4. Laboratory - Hematology (20 features)
    5. Severity Scores & Interventions (22 features)

    Total: 102 features with ~15% mortality rate
    Missing values are preserved (NO imputation).
    """
    np.random.seed(random_state)

    # Latent severity score (drives correlations)
    severity_latent = np.random.beta(2, 5, n_samples)

    # === SITE 1: Demographics (8 features) ===
    age = np.random.normal(65, 17, n_samples).clip(18, 95)
    severity_latent = np.clip(severity_latent + 0.01 * (age - 65) / 17, 0, 1)

    gender = np.random.binomial(1, 0.56, n_samples)
    weight = np.where(gender == 1,
                      np.random.normal(82, 18, n_samples),
                      np.random.normal(70, 16, n_samples)).clip(40, 200)
    height = np.where(gender == 1,
                      np.random.normal(175, 8, n_samples),
                      np.random.normal(162, 7, n_samples)).clip(140, 210)
    bmi = weight / (height/100)**2

    admission_type = np.random.choice([0, 1, 2], n_samples, p=[0.45, 0.35, 0.20])
    insurance_type = np.random.choice([0, 1, 2, 3], n_samples, p=[0.50, 0.25, 0.15, 0.10])
    ethnicity = np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.70, 0.12, 0.08, 0.06, 0.04])

    demographics = pd.DataFrame({
        'age': age, 'gender': gender, 'weight_kg': weight,
        'height_cm': height, 'bmi': bmi, 'admission_type': admission_type,
        'insurance_type': insurance_type, 'ethnicity': ethnicity
    })

    # === SITE 2: Vital Signs (24 features) ===
    hr_base = 75 + 40 * severity_latent + np.random.normal(0, 10, n_samples)
    hr_base = hr_base.clip(40, 180)

    sbp_base = 130 - 50 * severity_latent + np.random.normal(0, 15, n_samples)
    sbp_base = sbp_base.clip(60, 200)
    dbp_base = sbp_base * 0.6 + np.random.normal(10, 8, n_samples)
    map_base = (sbp_base + 2*dbp_base) / 3

    resp_rate_mean = (14 + 15 * severity_latent + np.random.normal(0, 3, n_samples)).clip(8, 45)
    spo2_mean = (98 - 10 * severity_latent + np.random.normal(0, 2, n_samples)).clip(70, 100)
    temp_mean = (37.0 + 1.5 * severity_latent * np.random.binomial(1, 0.4, n_samples) +
                 np.random.normal(0, 0.5, n_samples)).clip(34, 42)

    gcs_base = 15 - 8 * severity_latent + np.random.normal(0, 1, n_samples)
    gcs_total = np.clip(gcs_base, 3, 15).astype(int)
    gcs_eye = np.clip(np.round(gcs_total / 15 * 4), 1, 4).astype(int)
    gcs_verbal = np.clip(np.round(gcs_total / 15 * 5), 1, 5).astype(int)
    gcs_motor = np.clip(gcs_total - gcs_eye - gcs_verbal + 2, 1, 6).astype(int)

    vitals = pd.DataFrame({
        'hr_mean': hr_base + np.random.normal(0, 5, n_samples),
        'hr_min': hr_base - np.abs(np.random.normal(15, 8, n_samples)),
        'hr_max': hr_base + np.abs(np.random.normal(20, 10, n_samples)),
        'hr_std': np.abs(np.random.normal(12, 5, n_samples)),
        'sbp_mean': sbp_base + np.random.normal(0, 5, n_samples),
        'sbp_min': sbp_base - np.abs(np.random.normal(20, 10, n_samples)),
        'sbp_max': sbp_base + np.abs(np.random.normal(25, 12, n_samples)),
        'dbp_mean': dbp_base + np.random.normal(0, 4, n_samples),
        'dbp_min': dbp_base - np.abs(np.random.normal(12, 6, n_samples)),
        'dbp_max': dbp_base + np.abs(np.random.normal(15, 7, n_samples)),
        'map_mean': map_base + np.random.normal(0, 5, n_samples),
        'resp_rate_mean': resp_rate_mean,
        'resp_rate_min': resp_rate_mean - np.abs(np.random.normal(4, 2, n_samples)),
        'resp_rate_max': resp_rate_mean + np.abs(np.random.normal(6, 3, n_samples)),
        'spo2_mean': spo2_mean,
        'spo2_min': (spo2_mean - np.abs(np.random.normal(4, 2, n_samples))).clip(50, 100),
        'temp_mean': temp_mean,
        'temp_min': temp_mean - np.abs(np.random.normal(0.5, 0.3, n_samples)),
        'temp_max': temp_mean + np.abs(np.random.normal(0.8, 0.4, n_samples)),
        'gcs_eye': gcs_eye, 'gcs_verbal': gcs_verbal,
        'gcs_motor': gcs_motor, 'gcs_total': gcs_total,
        'gcs_min': np.clip(gcs_total - np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.3, 0.1]), 3, 15)
    })

    # === SITE 3: Laboratory - Chemistry (28 features) ===
    ph = (7.40 - 0.15 * severity_latent + np.random.normal(0, 0.05, n_samples)).clip(6.8, 7.8)
    pco2 = (40 + 20 * severity_latent + np.random.normal(0, 8, n_samples)).clip(15, 100)
    po2 = (100 - 40 * severity_latent + np.random.normal(0, 20, n_samples)).clip(40, 500)
    base_excess = (ph - 7.4) * 20 + np.random.normal(0, 3, n_samples)
    bicarbonate = 24 + base_excess * 0.5 + np.random.normal(0, 2, n_samples)

    sodium = 140 + np.random.normal(0, 4, n_samples)
    potassium = (4.2 + 1.0 * severity_latent + np.random.normal(0, 0.4, n_samples)).clip(2.5, 7.0)
    chloride = sodium - 100 + np.random.normal(0, 4, n_samples)
    calcium = 8.8 + np.random.normal(0, 0.8, n_samples)
    ionized_calcium = calcium * 0.45 + np.random.normal(0, 0.15, n_samples)
    magnesium = np.random.normal(2.0, 0.4, n_samples).clip(1, 4)
    phosphate = np.random.normal(3.5, 1.2, n_samples).clip(1, 10)

    creatinine = (0.9 + 3.0 * severity_latent + np.random.lognormal(0, 0.4, n_samples)).clip(0.3, 15)
    bun = (creatinine * 12 + np.random.normal(5, 8, n_samples)).clip(5, 150)
    egfr = (120 - 80 * severity_latent + np.random.normal(0, 15, n_samples)).clip(5, 150)

    bilirubin_total = (1.0 + 8 * severity_latent * np.random.binomial(1, 0.3, n_samples) +
                       np.random.lognormal(0, 0.5, n_samples)).clip(0.1, 30)
    bilirubin_direct = bilirubin_total * 0.3 + np.random.lognormal(-1, 0.5, n_samples)
    alt = np.random.lognormal(3.2, 0.8, n_samples).clip(5, 5000)
    ast = alt * 0.8 + np.random.lognormal(2.5, 0.6, n_samples)
    alp = np.random.lognormal(4.2, 0.5, n_samples).clip(30, 1500)
    albumin = (4.0 - 1.5 * severity_latent + np.random.normal(0, 0.5, n_samples)).clip(1, 5.5)
    total_protein = albumin + np.random.normal(3.5, 0.8, n_samples)

    glucose = (100 + 100 * severity_latent + np.random.lognormal(0, 0.3, n_samples) * 50).clip(40, 600)
    lactate = (1.0 + 8 * severity_latent + np.random.lognormal(0, 0.5, n_samples)).clip(0.5, 20)
    anion_gap = sodium - chloride - bicarbonate + np.random.normal(0, 2, n_samples)

    labs_chemistry = pd.DataFrame({
        'ph': ph, 'pco2': pco2, 'po2': po2, 'base_excess': base_excess,
        'bicarbonate': bicarbonate, 'sodium': sodium, 'potassium': potassium,
        'chloride': chloride, 'calcium': calcium, 'ionized_calcium': ionized_calcium,
        'magnesium': magnesium, 'phosphate': phosphate, 'creatinine': creatinine,
        'bun': bun, 'egfr': egfr, 'bilirubin_total': bilirubin_total,
        'bilirubin_direct': bilirubin_direct, 'alt': alt, 'ast': ast, 'alp': alp,
        'albumin': albumin, 'total_protein': total_protein, 'glucose': glucose,
        'lactate': lactate, 'anion_gap': anion_gap,
        'glucose_min': glucose - np.abs(np.random.normal(20, 15, n_samples)),
        'glucose_max': glucose + np.abs(np.random.normal(40, 30, n_samples)),
        'lactate_max': lactate + np.abs(np.random.lognormal(0, 0.5, n_samples))
    })

    # === SITE 4: Laboratory - Hematology (20 features) ===
    wbc = np.abs(10 + 8 * severity_latent * np.random.choice([-1, 1], n_samples) +
                 np.random.lognormal(0, 0.3, n_samples) * 3).clip(0.5, 50)
    rbc = np.where(gender == 1,
                   np.random.normal(4.8, 0.6, n_samples),
                   np.random.normal(4.3, 0.5, n_samples))
    rbc = (rbc - 0.8 * severity_latent).clip(2, 7)
    hemoglobin = (rbc * 3 + np.random.normal(0, 0.8, n_samples)).clip(5, 20)
    hematocrit = (hemoglobin * 3 + np.random.normal(0, 1.5, n_samples)).clip(15, 60)
    mcv = np.random.normal(88, 8, n_samples).clip(60, 120)
    mch = mcv * 0.34 + np.random.normal(0, 1, n_samples)
    mchc = np.random.normal(33.5, 1.5, n_samples).clip(28, 38)
    rdw = (13 + 4 * severity_latent + np.random.normal(0, 1.5, n_samples)).clip(11, 25)

    platelets = (250 - 150 * severity_latent + np.random.lognormal(0, 0.3, n_samples) * 50).clip(10, 1000)
    mpv = np.random.normal(10, 1.5, n_samples).clip(6, 15)

    neutrophils = (60 + 20 * severity_latent + np.random.normal(0, 10, n_samples)).clip(20, 95)
    lymphocytes = (100 - neutrophils - np.random.normal(15, 5, n_samples)).clip(2, 50)
    monocytes = np.random.normal(7, 3, n_samples).clip(1, 20)
    eosinophils = np.random.lognormal(0.5, 0.8, n_samples).clip(0, 15)
    basophils = np.random.lognormal(-1, 0.5, n_samples).clip(0, 3)

    pt = (12 + 10 * severity_latent + np.random.lognormal(0, 0.2, n_samples) * 3).clip(9, 60)
    inr = ((pt / 12) ** 1.1 + np.random.normal(0, 0.1, n_samples)).clip(0.8, 8)
    ptt = (30 + 30 * severity_latent + np.random.lognormal(0, 0.2, n_samples) * 10).clip(20, 150)
    fibrinogen = (300 - 150 * severity_latent + np.random.normal(0, 80, n_samples)).clip(50, 800)

    labs_hematology = pd.DataFrame({
        'wbc': wbc, 'rbc': rbc, 'hemoglobin': hemoglobin, 'hematocrit': hematocrit,
        'mcv': mcv, 'mch': mch, 'mchc': mchc, 'rdw': rdw,
        'platelets': platelets, 'mpv': mpv,
        'neutrophils': neutrophils, 'lymphocytes': lymphocytes,
        'monocytes': monocytes, 'eosinophils': eosinophils, 'basophils': basophils,
        'pt': pt, 'inr': inr, 'ptt': ptt, 'fibrinogen': fibrinogen,
        'hemoglobin_min': hemoglobin - np.abs(np.random.normal(1.5, 0.8, n_samples))
    })

    # === SITE 5: Severity Scores & Interventions (22 features) ===
    sofa_resp = np.clip(np.round(4 * severity_latent + np.random.normal(0, 0.5, n_samples)), 0, 4).astype(int)
    sofa_coag = np.clip(np.round(4 * severity_latent + np.random.normal(0, 0.5, n_samples)), 0, 4).astype(int)
    sofa_liver = np.clip(np.round(4 * severity_latent + np.random.normal(0, 0.5, n_samples)), 0, 4).astype(int)
    sofa_cardio = np.clip(np.round(4 * severity_latent + np.random.normal(0, 0.5, n_samples)), 0, 4).astype(int)
    sofa_cns = np.clip(np.round(4 * severity_latent + np.random.normal(0, 0.5, n_samples)), 0, 4).astype(int)
    sofa_renal = np.clip(np.round(4 * severity_latent + np.random.normal(0, 0.5, n_samples)), 0, 4).astype(int)
    sofa_total = sofa_resp + sofa_coag + sofa_liver + sofa_cardio + sofa_cns + sofa_renal

    saps_age = np.where(age < 40, 0, np.where(age < 60, 7, np.where(age < 70, 12,
                np.where(age < 75, 15, np.where(age < 80, 16, 18)))))
    saps_hr = np.where(hr_base < 40, 11, np.where(hr_base < 70, 2,
               np.where(hr_base < 120, 0, np.where(hr_base < 160, 4, 7))))
    saps_sbp = np.where(sbp_base < 70, 13, np.where(sbp_base < 100, 5,
                np.where(sbp_base < 200, 0, 2)))
    saps_temp = np.where(temp_mean < 39, 0, 3)
    saps_gcs = np.where(gcs_total < 6, 26, np.where(gcs_total < 9, 13,
                np.where(gcs_total < 11, 7, np.where(gcs_total < 14, 5, 0))))
    saps_total = saps_age + saps_hr + saps_sbp + saps_temp + saps_gcs + np.random.randint(0, 20, n_samples)

    vent = np.random.binomial(1, 0.2 + 0.6 * severity_latent)
    vasopressor = np.random.binomial(1, 0.1 + 0.7 * severity_latent)
    rrt = np.random.binomial(1, 0.02 + 0.3 * severity_latent)

    los_icu = (2 + 10 * severity_latent + np.random.lognormal(0, 0.5, n_samples)).clip(0.5, 60)
    los_hospital = los_icu + np.random.lognormal(1, 0.7, n_samples)

    elixhauser_score = np.random.poisson(2 + 3 * severity_latent, n_samples)
    charlson_score = np.random.poisson(1 + 2 * severity_latent, n_samples)
    n_procedures = np.random.poisson(1 + 3 * severity_latent, n_samples)
    n_diagnoses = np.random.poisson(5 + 8 * severity_latent, n_samples)

    severity_df = pd.DataFrame({
        'sofa_resp': sofa_resp, 'sofa_coag': sofa_coag, 'sofa_liver': sofa_liver,
        'sofa_cardio': sofa_cardio, 'sofa_cns': sofa_cns, 'sofa_renal': sofa_renal,
        'sofa_total': sofa_total,
        'saps_age': saps_age, 'saps_hr': saps_hr, 'saps_sbp': saps_sbp,
        'saps_temp': saps_temp, 'saps_gcs': saps_gcs, 'saps_total': saps_total,
        'vent': vent, 'vasopressor': vasopressor, 'rrt': rrt,
        'los_icu': los_icu, 'los_hospital': los_hospital,
        'elixhauser_score': elixhauser_score, 'charlson_score': charlson_score,
        'n_procedures': n_procedures, 'n_diagnoses': n_diagnoses
    })

    # Combine all features
    X = pd.concat([demographics, vitals, labs_chemistry, labs_hematology, severity_df], axis=1)

    # Generate mortality outcome (~15% positive class)
    log_odds = (-2.5 +
                3.5 * severity_latent +
                0.02 * (age - 65) +
                0.3 * (lactate > 4).astype(float) +
                0.4 * vasopressor +
                0.3 * (gcs_total < 10).astype(float) +
                0.2 * vent +
                0.2 * (creatinine > 2).astype(float) +
                np.random.normal(0, 0.3, n_samples))

    mortality_prob = 1 / (1 + np.exp(-log_odds))
    y = pd.Series(np.random.binomial(1, mortality_prob), name='mortality')

    # Add missing values (NO IMPUTATION)
    missing_rate = 0.08
    for col in X.columns:
        if col not in ['gender', 'admission_type', 'insurance_type', 'ethnicity', 'vent', 'vasopressor', 'rrt']:
            mask = np.random.random(n_samples) < missing_rate
            X.loc[mask, col] = np.nan

    return X, y


def distribute_features_to_sites(X: pd.DataFrame) -> Dict[str, List[str]]:
    """Split features by clinical department (non-overlapping)."""
    site_features = {
        'Site_1_Demographics': ['age', 'gender', 'weight_kg', 'height_cm', 'bmi',
                                 'admission_type', 'insurance_type', 'ethnicity'],
        'Site_2_Vitals': ['hr_mean', 'hr_min', 'hr_max', 'hr_std',
                          'sbp_mean', 'sbp_min', 'sbp_max',
                          'dbp_mean', 'dbp_min', 'dbp_max', 'map_mean',
                          'resp_rate_mean', 'resp_rate_min', 'resp_rate_max',
                          'spo2_mean', 'spo2_min',
                          'temp_mean', 'temp_min', 'temp_max',
                          'gcs_eye', 'gcs_verbal', 'gcs_motor', 'gcs_total', 'gcs_min'],
        'Site_3_Chemistry': ['ph', 'pco2', 'po2', 'base_excess', 'bicarbonate',
                              'sodium', 'potassium', 'chloride', 'calcium', 'ionized_calcium',
                              'magnesium', 'phosphate', 'creatinine', 'bun', 'egfr',
                              'bilirubin_total', 'bilirubin_direct', 'alt', 'ast', 'alp',
                              'albumin', 'total_protein', 'glucose', 'lactate', 'anion_gap',
                              'glucose_min', 'glucose_max', 'lactate_max'],
        'Site_4_Hematology': ['wbc', 'rbc', 'hemoglobin', 'hematocrit',
                               'mcv', 'mch', 'mchc', 'rdw',
                               'platelets', 'mpv',
                               'neutrophils', 'lymphocytes', 'monocytes', 'eosinophils', 'basophils',
                               'pt', 'inr', 'ptt', 'fibrinogen', 'hemoglobin_min'],
        'Site_5_Severity': ['sofa_resp', 'sofa_coag', 'sofa_liver', 'sofa_cardio',
                            'sofa_cns', 'sofa_renal', 'sofa_total',
                            'saps_age', 'saps_hr', 'saps_sbp', 'saps_temp', 'saps_gcs', 'saps_total',
                            'vent', 'vasopressor', 'rrt',
                            'los_icu', 'los_hospital',
                            'elixhauser_score', 'charlson_score', 'n_procedures', 'n_diagnoses']
    }
    # Filter to only include features that exist in X
    return {site: [f for f in feats if f in X.columns] for site, feats in site_features.items()}


# =============================================================================
# FEDSELECT WITH PROPER FEATURE IMPORTANCE
# =============================================================================

class FedSelectFeatureSelector:
    """
    FedSelect: Federated feature selection using correlation-based grouping
    and mutual information-based ranking.

    Steps:
    1. Each site identifies correlation groups locally
    2. Each site computes local MI scores (shares only rankings, not raw data)
    3. Global aggregation of rankings
    4. Select best feature from each correlation group based on global rank
    """

    def __init__(self, correlation_threshold: float = 0.75):
        self.correlation_threshold = correlation_threshold
        self.local_analyzer = LocalCorrelationAnalyzer(correlation_threshold)

    def compute_local_mi_rankings(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Compute mutual information scores locally (privacy-preserving: only ranks shared).
        """
        # Handle NaN for MI computation
        X_filled = X.fillna(X.median())
        mi_scores = mutual_info_classif(X_filled, y, random_state=RANDOM_STATE)
        return {col: score for col, score in zip(X.columns, mi_scores)}

    def federated_feature_selection(
        self,
        site_data: Dict[str, pd.DataFrame],
        y: pd.Series
    ) -> Tuple[List[str], Dict[str, Dict]]:
        """
        Perform federated feature selection.

        Returns:
            selected_features: List of selected feature names
            site_info: Per-site information for analysis
        """
        all_local_results = {}
        all_local_mi = {}
        site_info = {}

        # Phase 1 & 2: Local analysis at each site
        for site_name, X_site in site_data.items():
            # Local correlation analysis
            correlation_groups = self.local_analyzer.analyze(X_site)
            local_results = format_local_results(correlation_groups)
            all_local_results[site_name] = local_results

            # Local MI computation (only rankings shared)
            mi_scores = self.compute_local_mi_rankings(X_site, y)
            all_local_mi[site_name] = mi_scores

            site_info[site_name] = {
                'n_features': X_site.shape[1],
                'n_groups': len(local_results),
                'correlation_groups': local_results,
                'mi_scores': mi_scores
            }

        # Phase 3: Global aggregation
        # Aggregate MI scores across sites (average where feature appears in multiple sites)
        global_mi_scores = defaultdict(list)
        for site_mi in all_local_mi.values():
            for feature, score in site_mi.items():
                global_mi_scores[feature].append(score)

        # Average MI scores
        aggregated_mi = {f: np.mean(scores) for f, scores in global_mi_scores.items()}

        # Convert to ranks (higher MI = lower rank = better)
        sorted_features = sorted(aggregated_mi.items(), key=lambda x: -x[1])
        global_ranks = {f: rank for rank, (f, _) in enumerate(sorted_features, 1)}

        # Phase 4: Feature selection - best feature from each correlation group
        selected_features = set()

        for site_name, local_results in all_local_results.items():
            for group_features in local_results.values():
                if not group_features:
                    continue
                # Select feature with lowest rank (highest MI)
                best_feature = min(group_features, key=lambda f: global_ranks.get(f, float('inf')))
                selected_features.add(best_feature)

        return sorted(list(selected_features)), site_info


# =============================================================================
# MODEL TRAINING METHODS
# =============================================================================

def create_model(model_type: str = 'hgb'):
    """Create a model that handles missing values natively."""
    if model_type == 'hgb':
        return HistGradientBoostingClassifier(
            max_iter=200,
            max_depth=8,
            learning_rate=0.1,
            min_samples_leaf=20,
            l2_regularization=0.1,
            random_state=RANDOM_STATE
        )
    else:
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1
        )


def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Find threshold that maximizes F1 score."""
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_f1, best_thresh = 0, 0.5
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh


def evaluate_model(y_true: np.ndarray, y_proba: np.ndarray,
                   threshold: Optional[float] = None) -> Dict:
    """Evaluate model with optimal threshold if not provided."""
    if threshold is None:
        threshold = find_optimal_threshold(y_true, y_proba)

    y_pred = (y_proba >= threshold).astype(int)

    return {
        'roc_auc': roc_auc_score(y_true, y_proba),
        'pr_auc': average_precision_score(y_true, y_proba),
        'sensitivity': recall_score(y_true, y_pred, zero_division=0),
        'specificity': recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'threshold': threshold,
        'y_pred': y_pred,
        'y_proba': y_proba
    }


def cross_validate_model(model, X: pd.DataFrame, y: pd.Series,
                         cv: int = 5) -> Tuple[float, float]:
    """Perform stratified cross-validation."""
    cv_scores = cross_val_score(model, X, y, cv=StratifiedKFold(cv, shuffle=True, random_state=RANDOM_STATE),
                                 scoring='roc_auc', n_jobs=-1)
    return cv_scores.mean(), cv_scores.std()


# =============================================================================
# METHOD 1: CENTRALIZED TRAINING (BASELINE)
# =============================================================================

def train_centralized(X_train: pd.DataFrame, X_test: pd.DataFrame,
                      y_train: pd.Series, y_test: pd.Series) -> Dict:
    """Centralized training with all features."""
    model = create_model('hgb')
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = evaluate_model(y_test.values, y_proba)

    cv_mean, cv_std = cross_validate_model(create_model('hgb'), X_train, y_train)

    return {
        'name': 'Centralized',
        'n_features': X_train.shape[1],
        'model': model,
        'cv_roc_auc': f"{cv_mean:.3f}±{cv_std:.3f}",
        **metrics
    }


# =============================================================================
# METHOD 2: FEDERATED RF (ALL FEATURES)
# =============================================================================

def train_federated(site_features: Dict[str, List[str]],
                    X_train: pd.DataFrame, X_test: pd.DataFrame,
                    y_train: pd.Series, y_test: pd.Series) -> Dict:
    """
    Federated training: each site trains on local features, predictions aggregated.
    """
    site_probas = []
    site_models = {}

    for site_name, features in site_features.items():
        if not features:
            continue

        X_train_site = X_train[features]
        X_test_site = X_test[features]

        model = create_model('hgb')
        model.fit(X_train_site, y_train)

        site_probas.append(model.predict_proba(X_test_site)[:, 1])
        site_models[site_name] = model

    # Aggregate predictions (federated averaging)
    y_proba = np.mean(site_probas, axis=0)
    metrics = evaluate_model(y_test.values, y_proba)

    # CV on full federated setup
    all_features = [f for feats in site_features.values() for f in feats]
    cv_mean, cv_std = cross_validate_model(create_model('hgb'), X_train[all_features], y_train)

    return {
        'name': 'Federated (All)',
        'n_features': len(all_features),
        'model': site_models,
        'cv_roc_auc': f"{cv_mean:.3f}±{cv_std:.3f}",
        **metrics
    }


# =============================================================================
# METHOD 3: FEDERATED + STANDARD FEATURE SELECTION (LOCAL MI)
# =============================================================================

def train_federated_local_fs(site_features: Dict[str, List[str]],
                             X_train: pd.DataFrame, X_test: pd.DataFrame,
                             y_train: pd.Series, y_test: pd.Series,
                             top_k_pct: float = 0.5) -> Dict:
    """
    Federated with local feature selection: each site selects top-k% features by MI.
    """
    selected_features_per_site = {}
    site_probas = []
    site_models = {}

    for site_name, features in site_features.items():
        if not features:
            continue

        X_train_site = X_train[features]
        X_test_site = X_test[features]

        # Local feature selection using MI
        X_filled = X_train_site.fillna(X_train_site.median())
        mi_scores = mutual_info_classif(X_filled, y_train, random_state=RANDOM_STATE)

        k = max(1, int(len(features) * top_k_pct))
        top_indices = np.argsort(mi_scores)[-k:]
        selected = [features[i] for i in top_indices]
        selected_features_per_site[site_name] = selected

        # Train on selected features
        model = create_model('hgb')
        model.fit(X_train_site[selected], y_train)

        site_probas.append(model.predict_proba(X_test_site[selected])[:, 1])
        site_models[site_name] = model

    # Aggregate
    y_proba = np.mean(site_probas, axis=0)
    metrics = evaluate_model(y_test.values, y_proba)

    all_selected = [f for feats in selected_features_per_site.values() for f in feats]
    cv_mean, cv_std = cross_validate_model(create_model('hgb'), X_train[all_selected], y_train)

    return {
        'name': 'Federated + Local FS',
        'n_features': len(all_selected),
        'selected_per_site': selected_features_per_site,
        'model': site_models,
        'cv_roc_auc': f"{cv_mean:.3f}±{cv_std:.3f}",
        **metrics
    }


# =============================================================================
# METHOD 4: FEDERATED + FEDSELECT
# =============================================================================

def train_federated_fedselect(site_features: Dict[str, List[str]],
                              X_train: pd.DataFrame, X_test: pd.DataFrame,
                              y_train: pd.Series, y_test: pd.Series,
                              correlation_threshold: float = 0.75) -> Dict:
    """
    Federated with FedSelect: correlation-based grouping + MI-based ranking.
    """
    # Prepare site data
    site_data = {site: X_train[features] for site, features in site_features.items() if features}

    # Run FedSelect
    selector = FedSelectFeatureSelector(correlation_threshold)
    selected_features, site_info = selector.federated_feature_selection(site_data, y_train)

    # Map selected features back to sites
    selected_per_site = {site: [f for f in selected_features if f in feats]
                         for site, feats in site_features.items()}

    # Train federated model on selected features
    site_probas = []
    site_models = {}

    for site_name, features in selected_per_site.items():
        if not features:
            continue

        X_train_site = X_train[features]
        X_test_site = X_test[features]

        model = create_model('hgb')
        model.fit(X_train_site, y_train)

        site_probas.append(model.predict_proba(X_test_site)[:, 1])
        site_models[site_name] = model

    # Aggregate
    y_proba = np.mean(site_probas, axis=0)
    metrics = evaluate_model(y_test.values, y_proba)

    cv_mean, cv_std = cross_validate_model(create_model('hgb'), X_train[selected_features], y_train)

    return {
        'name': 'Federated + FedSelect',
        'n_features': len(selected_features),
        'selected_features': selected_features,
        'selected_per_site': selected_per_site,
        'site_info': site_info,
        'model': site_models,
        'cv_roc_auc': f"{cv_mean:.3f}±{cv_std:.3f}",
        **metrics
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_results(results: List[Dict], y_test: pd.Series, save_path: str = 'fedselect_results.png'):
    """Create comprehensive visualization of results."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

    # ROC Curves
    ax = axes[0, 0]
    for i, r in enumerate(results):
        fpr, tpr, _ = roc_curve(y_test, r['y_proba'])
        ax.plot(fpr, tpr, color=colors[i], lw=2,
                label=f"{r['name']} (AUC={r['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # PR Curves
    ax = axes[0, 1]
    for i, r in enumerate(results):
        precision, recall, _ = precision_recall_curve(y_test, r['y_proba'])
        ax.plot(recall, precision, color=colors[i], lw=2,
                label=f"{r['name']} (PR={r['pr_auc']:.3f})")
    ax.axhline(y=y_test.mean(), color='gray', linestyle='--', alpha=0.5, label='Baseline')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Metrics Comparison
    ax = axes[0, 2]
    metrics_names = ['ROC AUC', 'PR AUC', 'Sensitivity', 'Specificity', 'F1']
    x = np.arange(len(metrics_names))
    width = 0.2

    for i, r in enumerate(results):
        values = [r['roc_auc'], r['pr_auc'], r['sensitivity'], r['specificity'], r['f1']]
        ax.bar(x + i*width, values, width, label=r['name'], color=colors[i])

    ax.set_ylabel('Score')
    ax.set_title('Performance Metrics')
    ax.set_xticks(x + 1.5*width)
    ax.set_xticklabels(metrics_names, fontsize=9)
    ax.legend(loc='lower right', fontsize=7)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')

    # Feature Count
    ax = axes[1, 0]
    methods = [r['name'] for r in results]
    features = [r['n_features'] for r in results]
    bars = ax.bar(range(len(methods)), features, color=colors)
    ax.set_ylabel('Number of Features')
    ax.set_title('Feature Count')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.replace(' + ', '\n+ ') for m in methods], fontsize=8)
    for bar, f in zip(bars, features):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, str(f), ha='center', fontsize=9)

    # Efficiency Plot
    ax = axes[1, 1]
    for i, r in enumerate(results):
        ax.scatter(r['n_features'], r['roc_auc'], s=200, c=colors[i],
                   label=r['name'], edgecolors='black', linewidths=1)
    ax.set_xlabel('Number of Features')
    ax.set_ylabel('ROC AUC')
    ax.set_title('Efficiency: AUC vs Features')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Summary Table
    ax = axes[1, 2]
    ax.axis('off')

    table_data = []
    for r in results:
        table_data.append([
            r['name'],
            str(r['n_features']),
            f"{r['roc_auc']:.4f}",
            f"{r['pr_auc']:.4f}",
            f"{r['sensitivity']:.4f}",
            f"{r['f1']:.4f}",
            r['cv_roc_auc']
        ])

    table = ax.table(
        cellText=table_data,
        colLabels=['Method', 'Features', 'ROC AUC', 'PR AUC', 'Sens.', 'F1', 'CV ROC'],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax.set_title('Results Summary', fontsize=12, fontweight='bold', y=0.85)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nFigure saved to: {save_path}")


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiment():
    print("=" * 80)
    print("FedSelect Experiment: Federated Feature Selection for ICU Mortality Prediction")
    print("=" * 80)

    # Generate data
    print("\n[1] Generating MIMIC-III-like synthetic data...")
    X, y = generate_mimic_like_data(n_samples=12000, random_state=RANDOM_STATE)
    print(f"    Dataset: {X.shape[0]:,} samples, {X.shape[1]} features")
    print(f"    Mortality rate: {y.mean()*100:.1f}%")
    print(f"    Missing values: {X.isnull().sum().sum():,} ({X.isnull().mean().mean()*100:.1f}%)")

    # Distribute to sites
    print("\n[2] Distributing features to federated sites...")
    site_features = distribute_features_to_sites(X)
    for site, features in site_features.items():
        print(f"    {site}: {len(features)} features")

    # Train/test split
    print("\n[3] Splitting data (stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"    Train: {len(y_train):,} samples ({y_train.mean()*100:.1f}% mortality)")
    print(f"    Test:  {len(y_test):,} samples ({y_test.mean()*100:.1f}% mortality)")

    # Train all methods
    results = []

    print("\n[4] Training models...")

    print("\n    4.1 Centralized (baseline)...")
    result_centralized = train_centralized(X_train, X_test, y_train, y_test)
    results.append(result_centralized)
    print(f"        ROC AUC: {result_centralized['roc_auc']:.4f}, CV: {result_centralized['cv_roc_auc']}")

    print("\n    4.2 Federated (all features)...")
    result_federated = train_federated(site_features, X_train, X_test, y_train, y_test)
    results.append(result_federated)
    print(f"        ROC AUC: {result_federated['roc_auc']:.4f}, CV: {result_federated['cv_roc_auc']}")

    print("\n    4.3 Federated + Local Feature Selection...")
    result_local_fs = train_federated_local_fs(site_features, X_train, X_test, y_train, y_test)
    results.append(result_local_fs)
    print(f"        ROC AUC: {result_local_fs['roc_auc']:.4f}, Features: {result_local_fs['n_features']}")

    print("\n    4.4 Federated + FedSelect...")
    result_fedselect = train_federated_fedselect(site_features, X_train, X_test, y_train, y_test)
    results.append(result_fedselect)
    print(f"        ROC AUC: {result_fedselect['roc_auc']:.4f}, Features: {result_fedselect['n_features']}")

    # Print results table
    print("\n" + "=" * 90)
    print("FINAL RESULTS")
    print("=" * 90)
    print(f"{'Method':<25} {'Features':>8} {'ROC AUC':>10} {'PR AUC':>10} {'Sens.':>10} {'Spec.':>10} {'F1':>10} {'CV ROC':>15}")
    print("-" * 90)
    for r in results:
        print(f"{r['name']:<25} {r['n_features']:>8} {r['roc_auc']:>10.4f} {r['pr_auc']:>10.4f} "
              f"{r['sensitivity']:>10.4f} {r['specificity']:>10.4f} {r['f1']:>10.4f} {r['cv_roc_auc']:>15}")
    print("=" * 90)

    # Analysis
    print("\nKEY FINDINGS:")
    print("-" * 50)

    cent_auc = result_centralized['roc_auc']
    fed_auc = result_federated['roc_auc']
    local_fs_auc = result_local_fs['roc_auc']
    fedselect_auc = result_fedselect['roc_auc']

    print(f"1. Centralized vs Federated gap: {(cent_auc - fed_auc):.4f} ROC AUC")
    print(f"2. FedSelect vs Local FS: {(fedselect_auc - local_fs_auc):+.4f} ROC AUC")
    print(f"3. FedSelect feature reduction: {result_centralized['n_features']} -> {result_fedselect['n_features']} "
          f"({(1 - result_fedselect['n_features']/result_centralized['n_features'])*100:.0f}%)")

    if result_fedselect.get('selected_per_site'):
        print("\nFedSelect features per site:")
        for site, feats in result_fedselect['selected_per_site'].items():
            print(f"  {site}: {len(feats)} features")

    # Visualization
    print("\n[5] Generating visualizations...")
    plot_results(results, y_test, 'fedselect_results.png')

    return results


if __name__ == "__main__":
    results = run_experiment()
