import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates domain-specific Project Management features using row-wise
    transformations (safe from train-test data leakage).
    """
    data = df.copy()

    # 1. Cost Performance Index (CPI): < 1.0 indicates cost overrun
    data['cpi'] = data['project_budget'] / data['actual_cost']

    # 2. Schedule Performance Index (SPI): < 1.0 indicates schedule delay
    data['spi'] = data['estimated_timeline_months'] / data['actual_duration_months']

    # 3. Communication Complexity: PMBOK channels formula n(n-1)/2
    total_people = data['team_size'] + data['stakeholder_count']
    data['communication_complexity'] = (total_people * (total_people - 1)) / 2

    # 4. Methodology Mismatch Score
    # Penalizes rigid methodologies (Waterfall) when complexity and scope changes are high
    methodology_rigidity = data['methodology'].map({
        'Waterfall': 1.0,
        'Hybrid': 0.45,
        'Scrum': 0.15,
        'Agile': 0.10,
        'Kanban': 0.10
    }).fillna(0.25)

    data['methodology_mismatch_score'] = (
        methodology_rigidity
        * (data['complexity_score'] / 10.0)
        * data['change_request_frequency']
    ).round(3)

    return data


if __name__ == "__main__":
    raw_df = pd.read_csv("data/raw/project_management_data.csv")
    featured_df = engineer_features(raw_df)
    new_cols = ['cpi', 'spi', 'communication_complexity', 'methodology_mismatch_score']
    print("Engineered Features Preview:")
    print(featured_df[new_cols].describe().round(3))