import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor
)
from sklearn.metrics import mean_absolute_error, r2_score


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Project Cost Estimator",
    page_icon="◈",
    layout="wide"
)


# ============================================================
# LOAD DATASET
# ============================================================

try:
    df = pd.read_csv("Desharnais (2).csv")
except:
    try:
        df = pd.read_csv("Desharnais.csv")
    except:
        st.error(
            "Desharnais CSV file not found. "
            "Keep the CSV file in the same folder as app.py."
        )
        st.stop()


# ============================================================
# CHECK DATASET
# ============================================================

required_columns = [
    "id",
    "Project",
    "TeamExp",
    "ManagerExp",
    "YearEnd",
    "Length",
    "Effort",
    "Transactions",
    "Entities",
    "PointsNonAdjust",
    "Adjustment",
    "PointsAjust",
    "Language"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error("Missing columns in CSV:")
    st.write(missing_columns)
    st.stop()


# ============================================================
# PAGE TITLE
# ============================================================

st.title("◈ AI Project Cost Estimator")

st.write(
    "Machine Learning based effort and cost estimation "
    "using historical software project data."
)

st.info(
    "Historical data → Machine Learning → Effort prediction → "
    "Cost estimation → Scenario planning"
)


# ============================================================
# PREPARE DATA
# ============================================================

feature_columns = [
    "TeamExp",
    "ManagerExp",
    "YearEnd",
    "Length",
    "Transactions",
    "Entities",
    "PointsNonAdjust",
    "Adjustment",
    "PointsAjust",
    "Language"
]

X = df[feature_columns].copy()
y = df["Effort"].copy()


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ============================================================
# MODELS
# ============================================================

random_forest = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

gradient_boosting = GradientBoostingRegressor(
    random_state=42
)

extra_trees = ExtraTreesRegressor(
    n_estimators=100,
    random_state=42
)


# ============================================================
# TRAIN MODELS
# ============================================================

random_forest.fit(X_train, y_train)

gradient_boosting.fit(X_train, y_train)

extra_trees.fit(X_train, y_train)


# ============================================================
# PREDICTIONS
# ============================================================

rf_predictions = random_forest.predict(X_test)

gb_predictions = gradient_boosting.predict(X_test)

et_predictions = extra_trees.predict(X_test)


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(actual, predicted):

    actual = np.array(actual)
    predicted = np.array(predicted)

    mae = mean_absolute_error(actual, predicted)

    r2 = r2_score(actual, predicted)

    non_zero = actual != 0

    relative_error = np.abs(
        (actual[non_zero] - predicted[non_zero])
        / actual[non_zero]
    )

    mmre = np.mean(relative_error)

    mdmre = np.median(relative_error)

    pred25 = np.mean(relative_error <= 0.25) * 100

    return mae, r2, mmre, mdmre, pred25


# ============================================================
# MODEL METRICS
# ============================================================

rf_mae, rf_r2, rf_mmre, rf_mdmre, rf_pred25 = calculate_metrics(
    y_test,
    rf_predictions
)

gb_mae, gb_r2, gb_mmre, gb_mdmre, gb_pred25 = calculate_metrics(
    y_test,
    gb_predictions
)

et_mae, et_r2, et_mmre, et_mdmre, et_pred25 = calculate_metrics(
    y_test,
    et_predictions
)


# ============================================================
# CROSS VALIDATION
# ============================================================

rf_cv_mae = -cross_val_score(
    random_forest,
    X,
    y,
    cv=5,
    scoring="neg_mean_absolute_error"
).mean()

rf_cv_r2 = cross_val_score(
    random_forest,
    X,
    y,
    cv=5,
    scoring="r2"
).mean()


gb_cv_mae = -cross_val_score(
    gradient_boosting,
    X,
    y,
    cv=5,
    scoring="neg_mean_absolute_error"
).mean()

gb_cv_r2 = cross_val_score(
    gradient_boosting,
    X,
    y,
    cv=5,
    scoring="r2"
).mean()


et_cv_mae = -cross_val_score(
    extra_trees,
    X,
    y,
    cv=5,
    scoring="neg_mean_absolute_error"
).mean()

et_cv_r2 = cross_val_score(
    extra_trees,
    X,
    y,
    cv=5,
    scoring="r2"
).mean()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Project Inputs")

st.sidebar.subheader("Team Information")


# Team experience unit
team_exp_unit = st.sidebar.selectbox(
    "Team Experience Unit",
    ["Years", "Months"]
)

team_exp_value = st.sidebar.number_input(
    "Team Experience",
    min_value=0.0,
    value=2.0,
    step=0.5
)

if team_exp_unit == "Months":
    team_exp_years = team_exp_value / 12
else:
    team_exp_years = team_exp_value


# Manager experience unit
manager_exp_unit = st.sidebar.selectbox(
    "Manager Experience Unit",
    ["Years", "Months"]
)

manager_exp_value = st.sidebar.number_input(
    "Manager Experience",
    min_value=0.0,
    value=3.0,
    step=0.5
)

if manager_exp_unit == "Months":
    manager_exp_years = manager_exp_value / 12
else:
    manager_exp_years = manager_exp_value


# Team size
team_size = st.sidebar.number_input(
    "Team Size",
    min_value=1,
    max_value=500,
    value=10,
    step=1
)


st.sidebar.subheader("Project Information")


# Project year
project_year = st.sidebar.number_input(
    "Project Year",
    min_value=80,
    max_value=100,
    value=86,
    step=1
)


# Duration unit
duration_unit = st.sidebar.selectbox(
    "Project Duration Unit",
    ["Months", "Years"]
)

duration_value = st.sidebar.number_input(
    "Project Duration",
    min_value=0.1,
    value=6.0,
    step=0.5
)

if duration_unit == "Years":
    project_length_months = duration_value * 12
else:
    project_length_months = duration_value


transactions = st.sidebar.number_input(
    "Transactions",
    min_value=1,
    value=150,
    step=1
)


entities = st.sidebar.number_input(
    "Entities",
    min_value=1,
    value=80,
    step=1
)


function_points = st.sidebar.number_input(
    "Function Points",
    min_value=1,
    value=250,
    step=1
)


adjustment = st.sidebar.number_input(
    "Adjustment",
    min_value=0.0,
    value=25.0,
    step=1.0
)


adjusted_function_points = st.sidebar.number_input(
    "Adjusted Function Points",
    min_value=1,
    value=250,
    step=1
)


# Programming language
language_values = sorted(
    pd.to_numeric(
        df["Language"],
        errors="coerce"
    ).dropna().unique().tolist()
)

language = st.sidebar.selectbox(
    "Programming Language Code",
    language_values
)


st.sidebar.subheader("Cost Information")


hourly_rate = st.sidebar.number_input(
    "Cost per Person-Hour (₹)",
    min_value=1.0,
    value=500.0,
    step=50.0
)


tool_cost = st.sidebar.number_input(
    "Tool / Software Cost (₹)",
    min_value=0.0,
    value=0.0,
    step=1000.0
)


cloud_cost = st.sidebar.number_input(
    "Cloud / Infrastructure Cost (₹)",
    min_value=0.0,
    value=0.0,
    step=1000.0
)


# Optional actual effort
actual_effort_input = st.sidebar.number_input(
    "Actual Effort (optional)",
    min_value=0.0,
    value=0.0,
    step=100.0
)


# ============================================================
# NEW PROJECT DATA
# ============================================================

new_project = pd.DataFrame(
    [[
        team_exp_years,
        manager_exp_years,
        project_year,
        project_length_months,
        transactions,
        entities,
        function_points,
        adjustment,
        adjusted_function_points,
        language
    ]],
    columns=feature_columns
)


# ============================================================
# FINAL MODEL = EXTRA TREES
# ============================================================

predicted_effort = extra_trees.predict(
    new_project
)[0]


# ============================================================
# COST CALCULATIONS
# ============================================================

personnel_cost = predicted_effort * hourly_rate

total_cost = (
    personnel_cost
    + tool_cost
    + cloud_cost
)


# ============================================================
# ESTIMATED CALENDAR DURATION
# ============================================================

productive_hours_per_person_month = 160

estimated_months = predicted_effort / (
    team_size * productive_hours_per_person_month
)

estimated_years = estimated_months / 12


# ============================================================
# MAIN ESTIMATION RESULT
# ============================================================

st.header("Estimation Result")


col1, col2, col3 = st.columns(3)


with col1:
    st.metric(
        "Predicted Effort",
        f"{predicted_effort:,.0f} person-hours"
    )


with col2:
    st.metric(
        "Personnel Cost",
        f"₹{personnel_cost:,.0f}"
    )


with col3:
    st.metric(
        "Estimated Total Cost",
        f"₹{total_cost:,.0f}"
    )


st.write(
    f"With a team of **{team_size} people**, the estimated "
    f"calendar duration is approximately **{estimated_months:.1f} months**."
)


# ============================================================
# COST BREAKDOWN
# ============================================================

st.header("Cost Breakdown")


cost_data = pd.DataFrame({
    "Cost Type": [
        "Personnel",
        "Tools / Software",
        "Cloud / Infrastructure"
    ],
    "Cost": [
        personnel_cost,
        tool_cost,
        cloud_cost
    ]
})


st.bar_chart(
    cost_data.set_index("Cost Type")
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.header("Model Performance")


metric1, metric2, metric3, metric4, metric5 = st.columns(5)


with metric1:
    st.metric(
        "MAE",
        f"{et_mae:,.0f}"
    )


with metric2:
    st.metric(
        "R²",
        f"{et_r2:.3f}"
    )


with metric3:
    st.metric(
        "MMRE",
        f"{et_mmre:.3f}"
    )


with metric4:
    st.metric(
        "MdMRE",
        f"{et_mdmre:.3f}"
    )


with metric5:
    st.metric(
        "PRED(25)",
        f"{et_pred25:.1f}%"
    )


st.caption(
    "Lower MAE, MMRE and MdMRE are better. "
    "Higher R² and PRED(25) are better."
)


# ============================================================
# ACTUAL VS ESTIMATED
# ============================================================

st.header("Actual vs Estimated Effort")


comparison = pd.DataFrame({
    "Actual Effort": y_test.values,
    "Estimated Effort": et_predictions
})

comparison.index = range(1, len(comparison) + 1)

st.line_chart(comparison)


st.write(
    "The graph compares the actual effort from the test dataset "
    "with the effort predicted by the Extra Trees model."
)


# ============================================================
# MODEL COMPARISON
# ============================================================

st.header("Model Comparison")


model_comparison = pd.DataFrame({
    "Model": [
        "Random Forest",
        "Gradient Boosting",
        "Extra Trees"
    ],
    "MAE": [
        rf_mae,
        gb_mae,
        et_mae
    ],
    "R²": [
        rf_r2,
        gb_r2,
        et_r2
    ],
    "MMRE": [
        rf_mmre,
        gb_mmre,
        et_mmre
    ],
    "MdMRE": [
        rf_mdmre,
        gb_mdmre,
        et_mdmre
    ],
    "PRED(25) %": [
        rf_pred25,
        gb_pred25,
        et_pred25
    ]
})


st.dataframe(
    model_comparison.round(3),
    use_container_width=True,
    hide_index=True
)


st.subheader("Model MAE Comparison")


st.bar_chart(
    model_comparison.set_index("Model")[["MAE"]]
)


# ============================================================
# CROSS VALIDATION
# ============================================================

st.header("5-Fold Cross-Validation")


cv_comparison = pd.DataFrame({
    "Model": [
        "Random Forest",
        "Gradient Boosting",
        "Extra Trees"
    ],
    "CV MAE": [
        rf_cv_mae,
        gb_cv_mae,
        et_cv_mae
    ],
    "CV R²": [
        rf_cv_r2,
        gb_cv_r2,
        et_cv_r2
    ]
})


st.dataframe(
    cv_comparison.round(3),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# ESTIMATION UNCERTAINTY
# ============================================================

st.header("Prediction Uncertainty")


test_errors = np.abs(
    y_test.values - et_predictions
)

error_percentiles = np.percentile(
    test_errors,
    [25, 50, 75]
)

lower_effort = max(
    0,
    predicted_effort - error_percentiles[2]
)

higher_effort = (
    predicted_effort + error_percentiles[2]
)


uncertainty_data = pd.DataFrame({
    "Estimate": [
        "Lower estimate",
        "Expected estimate",
        "Higher estimate"
    ],
    "Effort (person-hours)": [
        lower_effort,
        predicted_effort,
        higher_effort
    ]
})


st.table(
    uncertainty_data
)


st.caption(
    "The range is derived from historical test-set prediction errors "
    "and is an empirical prediction range, not a statistically calibrated confidence interval."
)


# ============================================================
# SCENARIO PLANNING
# ============================================================

st.header("Scenario Planning")


optimistic_effort = predicted_effort * 0.80

expected_effort = predicted_effort

pessimistic_effort = predicted_effort * 1.20


optimistic_cost = (
    optimistic_effort * hourly_rate
    + tool_cost
    + cloud_cost
)

expected_cost = total_cost

pessimistic_cost = (
    pessimistic_effort * hourly_rate
    + tool_cost
    + cloud_cost
)


scenario_data = pd.DataFrame({
    "Scenario": [
        "Optimistic",
        "Expected",
        "Pessimistic"
    ],
    "Effort (person-hours)": [
        optimistic_effort,
        expected_effort,
        pessimistic_effort
    ],
    "Total Cost (₹)": [
        optimistic_cost,
        expected_cost,
        pessimistic_cost
    ]
})


st.dataframe(
    scenario_data.round(2),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# ESTIMATION BIAS
# ============================================================

st.header("Estimation Bias")


bias = np.mean(
    et_predictions - y_test.values
)


if bias > 0:
    st.write(
        f"The model has an average overestimation bias of "
        f"**{bias:,.0f} person-hours** on the test set."
    )

elif bias < 0:
    st.write(
        f"The model has an average underestimation bias of "
        f"**{abs(bias):,.0f} person-hours** on the test set."
    )

else:
    st.write(
        "The model has approximately zero average estimation bias."
    )


# ============================================================
# ACTUAL EFFORT CHECK
# ============================================================

if actual_effort_input > 0:

    st.header("Estimate vs Actual")

    difference = predicted_effort - actual_effort_input

    percentage_difference = (
        abs(difference) / actual_effort_input
    ) * 100

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Actual Effort",
            f"{actual_effort_input:,.0f} person-hours"
        )

    with col2:
        st.metric(
            "Estimated Effort",
            f"{predicted_effort:,.0f} person-hours"
        )

    with col3:
        st.metric(
            "Difference",
            f"{percentage_difference:.1f}%"
        )


# ============================================================
# HISTORICAL DATA COVERAGE
# ============================================================

st.header("Historical Data Coverage")


coverage1, coverage2, coverage3 = st.columns(3)


with coverage1:
    st.metric(
        "Historical Projects",
        len(df)
    )


with coverage2:
    st.metric(
        "Features Used",
        len(feature_columns)
    )


with coverage3:
    st.metric(
        "Test Projects",
        len(X_test)
    )


# ============================================================
# SIMILAR HISTORICAL PROJECTS
# ============================================================

st.header("Historical Project Reference")


numeric_features = [
    "TeamExp",
    "ManagerExp",
    "YearEnd",
    "Length",
    "Transactions",
    "Entities",
    "PointsNonAdjust",
    "Adjustment",
    "PointsAjust",
    "Language"
]


# Standardize values for distance calculation
reference_data = df[numeric_features].copy()

mean_values = reference_data.mean()

std_values = reference_data.std().replace(0, 1)

standardized_reference = (
    reference_data - mean_values
) / std_values

standardized_new = (
    new_project[numeric_features].iloc[0]
    - mean_values
) / std_values


distances = np.sqrt(
    (
        standardized_reference
        - standardized_new
    ).pow(2).sum(axis=1)
)


closest_indices = distances.nsmallest(5).index


similar_projects = df.loc[
    closest_indices,
    [
        "Project",
        "Effort",
        "Length",
        "TeamExp",
        "ManagerExp",
        "Transactions",
        "Entities"
    ]
].copy()


st.dataframe(
    similar_projects,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# PROJECT SUMMARY
# ============================================================

st.header("Project Summary")


summary = pd.DataFrame({
    "Parameter": [
        "Team Experience",
        "Manager Experience",
        "Team Size",
        "Project Year",
        "Project Duration",
        "Transactions",
        "Entities",
        "Function Points",
        "Adjustment",
        "Adjusted Function Points",
        "Programming Language",
        "Predicted Effort",
        "Personnel Cost",
        "Tool / Software Cost",
        "Cloud / Infrastructure Cost",
        "Total Estimated Cost"
    ],

    "Value": [
        f"{team_exp_value:g} {team_exp_unit.lower()}",
        f"{manager_exp_value:g} {manager_exp_unit.lower()}",
        f"{team_size} people",
        project_year,
        f"{duration_value:g} {duration_unit.lower()}",
        transactions,
        entities,
        function_points,
        adjustment,
        adjusted_function_points,
        language,
        f"{predicted_effort:,.0f} person-hours",
        f"₹{personnel_cost:,.0f}",
        f"₹{tool_cost:,.0f}",
        f"₹{cloud_cost:,.0f}",
        f"₹{total_cost:,.0f}"
    ]
})


st.dataframe(
    summary,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# AI ESTIMATION INSIGHTS
# ============================================================

st.header("Estimation Insights")


st.write(
    f"• The selected Extra Trees model predicts "
    f"**{predicted_effort:,.0f} person-hours** of effort."
)

st.write(
    f"• With a team of **{team_size} people**, the estimated "
    f"calendar duration is approximately **{estimated_months:.1f} months**."
)

st.write(
    f"• Personnel cost is estimated at "
    f"**₹{personnel_cost:,.0f}**."
)

st.write(
    f"• Including tools and cloud infrastructure, the total estimated "
    f"cost is **₹{total_cost:,.0f}**."
)

st.write(
    f"• Among the three tested models, Extra Trees has a test MAE of "
    f"**{et_mae:,.0f} person-hours**."
)

st.write(
    f"• PRED(25) for Extra Trees is **{et_pred25:.1f}%**, meaning "
    f"this percentage of test predictions fall within 25% of actual effort."
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Project Cost Estimation Prototype | "
    "Desharnais Dataset | Extra Trees Regression"
)