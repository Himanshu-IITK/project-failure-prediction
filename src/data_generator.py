import os
import numpy as np
import pandas as pd

def generate_project_dataset(n_samples: int = 2500, random_state: int = 42) -> pd.DataFrame:
    """
    Synthesizes a multi-domain Project Management dataset with realistic
    correlations, missing values, outliers, and latent failure patterns.
    """
    np.random.seed(random_state)

    # 1. Domain & Categorical Attributes
    domains = np.random.choice(
        ['Software Engineering', 'Agile IT', 'Enterprise Infrastructure', 'Startup Product'],
        size=n_samples,
        p=[0.35, 0.30, 0.20, 0.15]
    )
    
    methodologies = np.random.choice(
        ['Agile', 'Scrum', 'Kanban', 'Waterfall', 'Hybrid'],
        size=n_samples,
        p=[0.30, 0.25, 0.10, 0.20, 0.15]
    )

    # 2. Complexity & Team Attributes
    complexity_score = np.random.uniform(1.0, 10.0, size=n_samples).round(2)
    team_experience_years = np.random.gamma(shape=3.0, scale=1.8, size=n_samples).clip(0.5, 15.0).round(1)
    
    # Team size & stakeholders correlate positively with project complexity
    team_size = np.random.poisson(lam=4 + 1.8 * complexity_score).clip(3, 50)
    stakeholder_count = np.random.poisson(lam=2 + 0.9 * complexity_score + 0.15 * team_size).clip(2, 30)

    # 3. Schedule & Financial Baselines
    estimated_timeline_months = (
        np.random.uniform(2, 6, size=n_samples) + 1.5 * complexity_score
    ).clip(2, 36).round(1)

    # Budget scales with timeline, team size, and complexity (in USD)
    base_monthly_burn = team_size * np.random.uniform(4500, 8500, size=n_samples)
    project_budget = (base_monthly_burn * estimated_timeline_months).round(2)

    # 4. Execution Dynamics & Scope Creep
    change_request_frequency = np.random.poisson(lam=0.8 * complexity_score).clip(0, 25)
    defect_density = (
        np.random.exponential(scale=1.2, size=n_samples) * (complexity_score / 4.0)
    ).round(2)
    
    # Communication sentiment score (-1.0 = toxic/blocked, +1.0 = healthy/aligned)
    team_sentiment_score = np.random.normal(
        loc=0.25 - 0.03 * change_request_frequency, scale=0.35, size=n_samples
    ).clip(-1.0, 1.0).round(2)

    # 5. Actual Cost & Actual Schedule (Driven by risks, scope creep, and experience)
    cost_overrun_factor = (
        1.0
        + 0.035 * (complexity_score - 5.0)
        + 0.025 * change_request_frequency
        - 0.020 * (team_experience_years - 4.0)
        + np.random.normal(0, 0.12, size=n_samples)
    ).clip(0.65, 2.5)
    actual_cost = (project_budget * cost_overrun_factor).round(2)

    schedule_slip_factor = (
        1.0
        + 0.030 * (complexity_score - 5.0)
        + 0.020 * change_request_frequency
        + 0.006 * team_size
        - 0.025 * (team_experience_years - 4.0)
        + np.random.normal(0, 0.12, size=n_samples)
    ).clip(0.70, 2.4)
    actual_duration_months = (estimated_timeline_months * schedule_slip_factor).round(1)

    # 6. Determine Project Failure Target (0 = Success, 1 = Failure)
    # Latent risk combines budget overrun, schedule slip, methodology mismatch, and sentiment
    is_rigid_methodology = (methodologies == 'Waterfall').astype(int)
    mismatch_penalty = is_rigid_methodology * (complexity_score > 6.5).astype(int) * (change_request_frequency > 6).astype(int)

    latent_failure_logit = (
        -2.2
        + 3.2 * (cost_overrun_factor - 1.05)
        + 2.8 * (schedule_slip_factor - 1.05)
        + 0.12 * change_request_frequency
        - 0.90 * team_sentiment_score
        - 0.15 * team_experience_years
        + 1.35 * mismatch_penalty
        + 0.08 * defect_density
        + np.random.normal(0, 0.55, size=n_samples)
    )
    failure_probability = 1 / (1 + np.exp(-latent_failure_logit))
    project_failed = (np.random.uniform(0, 1, size=n_samples) < failure_probability).astype(int)

    # Assemble DataFrame
    df = pd.DataFrame({
        'project_id': [f"PRJ-{10000 + i}" for i in range(n_samples)],
        'domain': domains,
        'methodology': methodologies,
        'project_budget': project_budget,
        'actual_cost': actual_cost,
        'estimated_timeline_months': estimated_timeline_months,
        'actual_duration_months': actual_duration_months,
        'complexity_score': complexity_score,
        'team_size': team_size,
        'team_experience_years': team_experience_years,
        'stakeholder_count': stakeholder_count,
        'change_request_frequency': change_request_frequency,
        'defect_density': defect_density,
        'team_sentiment_score': team_sentiment_score,
        'project_failed': project_failed
    })

    # 7. Inject Realistic Imperfections (Missing Values & Outliers for Phase 4)
    for col in ['actual_cost', 'team_sentiment_score', 'change_request_frequency', 'defect_density']:
        missing_idx = np.random.choice(df.index, size=int(0.03 * n_samples), replace=False)
        df.loc[missing_idx, col] = np.nan

    # Inject extreme budget/cost outliers in 1% of projects
    outlier_idx = np.random.choice(df.index, size=int(0.01 * n_samples), replace=False)
    df.loc[outlier_idx, 'actual_cost'] = df.loc[outlier_idx, 'project_budget'] * np.random.uniform(4.0, 6.5)

    return df


if __name__ == "__main__":
    output_path = os.path.join("data", "raw", "project_management_data.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print("Synthesizing multi-domain project dataset...")
    dataset = generate_project_dataset(n_samples=2500)
    dataset.to_csv(output_path, index=False)
    
    print(f"Dataset saved successfully to: {output_path}")
    print(f"Total Rows: {len(dataset)} | Total Columns: {len(dataset.columns)}")
    print(f"Class Distribution (0 = Success, 1 = Failure):\n{dataset['project_failed'].value_counts(normalize=True).round(3)}")
    print(f"\nMissing Values per Column:\n{dataset.isnull().sum()[dataset.isnull().sum() > 0]}")