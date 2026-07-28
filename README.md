# Global Country Rankings: Exploratory Data Analysis and Predictive Modeling (2000–2026)

## Overview

This repository presents an exploratory data analysis (EDA) and a supervised machine learning study of a cross-national panel dataset covering **217 sovereign nations** over the period **2000–2026** (5,859 country-year observations). The dataset aggregates eleven internationally recognized global ranking indices — spanning happiness, human development, governance, economic wellbeing, and environmental performance — alongside categorical metadata (country, year, region, and economic tier).

The project was developed as an extension of an internship (staj) report and consists of two components:

1. **Exploratory Data Analysis** (`global_country_analysis.py`) — descriptive statistics, correlation analysis, and visualization of relationships between socioeconomic and governance indicators.
2. **Predictive Modeling** (`global_country_ml.py`) — a multiple linear regression model trained to estimate a country's `Happiness_Rank` from the remaining ranking indicators.

## Dataset Description

**Source file:** `dataset/global_country_rankings_2000_2026.csv`
**Original source:** [Global Country Rankings Real Database (2000–2026)](https://www.kaggle.com/datasets/ashyou09/global-country-indices-real-database-2000-2026), Kaggle
**Dimensions:** 5,859 rows × 15 columns
**Missing values:** none (fully populated panel)
**Duplicate records:** none

| Column | Description |
|---|---|
| `Country` | Standardized official country name (217 countries) |
| `Year` | Observation year, 2000–2026 |
| `Region` | Macro-region (North America, South America, Europe, Asia, Africa, Oceania) |
| `Economic_Tier` | Income classification (1 = High Income, 2 = Upper-Middle, 3 = Lower-Middle, 4 = Low Income) |
| `Happiness_Rank` | Global rank, World Happiness Report (1 = happiest) |
| `Global_Hunger_Rank` | Global rank, Global Hunger Index (1 = lowest hunger) |
| `Human_Development_Rank` | Global rank, UN Human Development Index (1 = highest development) |
| `GDP_Per_Capita_Rank` | Global rank, World Bank GDP per capita (1 = highest wealth) |
| `Life_Expectancy_Rank` | Global rank, World Bank life expectancy (1 = longest life) |
| `Corruption_Perception_Rank` | Global rank, Corruption Perceptions Index (1 = least corrupt) |
| `Democracy_Rank` | Global rank, EIU Democracy Index (1 = most democratic) |
| `Gini_Rank` | Global rank, World Bank Gini coefficient (1 = most equal) |
| `Press_Freedom_Rank` | Global rank, Reporters Without Borders (1 = most free press) |
| `Global_Peace_Rank` | Global rank, Global Peace Index (1 = most peaceful) |
| `Environmental_Performance_Rank` | Global rank, Yale Environmental Performance Index (1 = healthiest environment) |

All indicators except `Country`, `Year`, `Region`, and `Economic_Tier` are ordinal ranks in the range [1, 217], with lower values indicating better standing.

## Methodology

### 1. Exploratory Data Analysis

Data integrity was first verified through structural inspection (`head`, `tail`, `describe`, `info`, null- and duplicate-checks), confirming a clean panel with no preprocessing required. The analysis then proceeded along three lines:

- **Correlation structure** among all fifteen numeric indicators, visualized as a heatmap.
- **Distributional analysis** by categorical grouping variables (`Region`, `Economic_Tier`), using bar and box plots.
- **Pairwise relationships** between selected rank indicators, visualized as scatter plots with third-degree polynomial trend curves (fitted via `numpy.polyfit`) to capture non-linear association patterns.

### 2. Predictive Modeling

`Happiness_Rank` was set as the target variable, with all remaining numeric indicators (excluding `Country`, `Region`, and `Gini_Rank`) used as predictors. `Gini_Rank` was excluded after correlation analysis indicated negligible linear association with the other variables. Features were standardized using `StandardScaler` (z-score normalization), and the dataset was split into training and test sets (80/20, `random_state=42`). An ordinary least squares linear regression model (`sklearn.linear_model.LinearRegression`) was then fitted to the training data and evaluated on the held-out test set using R², MAE, and RMSE.

## Results

### Correlation Structure

<img src="data analysis plots/correlation_heatmap_analysis.png" alt="Correlation heatmap of all ranking indicators" width="700">

The heatmap reveals strong positive correlations among `Happiness_Rank`, `Human_Development_Rank`, `GDP_Per_Capita_Rank`, and `Life_Expectancy_Rank`, and strong correlations among the governance-related indicators (`Democracy_Rank`, `Press_Freedom_Rank`, `Corruption_Perception_Rank`, `Global_Peace_Rank`). `Gini_Rank` stands out as the indicator with the weakest association to the rest of the matrix, motivating its exclusion from the regression model.

### Regional Distribution

<img src="data analysis plots/Analysis_by_Regions.png" alt="Number of dataset entries by region" width="600">

Since each country contributes an equal number of yearly observations (2000–2026), the entry count per region directly reflects the number of countries per region in the dataset, with Africa and Europe contributing the largest number of country-year records.

### Human Development and Life Expectancy

<img src="data analysis plots/Analysis_of_Human_Development_Rank_by_Life_Expectancy_Rank.png" alt="Human Development Rank vs Life Expectancy Rank" width="600">

A near-monotonic positive relationship is observed between `Human_Development_Rank` and `Life_Expectancy_Rank`, consistent with life expectancy being a direct input component of the HDI methodology.

### Economic Tier and Democratic Governance

<img src="data analysis plots/Analysis_of_Economic_Tier_by_Democracy_Rank.png" alt="Economic Tier vs Democracy Rank" width="600">

Box-plot analysis shows that median `Democracy_Rank` deteriorates (increases numerically) as `Economic_Tier` moves from High Income (1) toward Low Income (4), with wider interquartile spread in the lower-income tiers, indicating greater heterogeneity of governance outcomes among poorer economies.

### Predictive Model: Actual vs. Predicted Happiness Rank

<img src="ml plots/plot_of_test_vs_predicted.png" alt="Predicted vs actual Happiness Rank on the test set" width="600">

| Metric | Value |
|---|---|
| R² | 1.0000 |
| MAE | ≈ 6.49 × 10⁻¹⁴ |
| RMSE | ≈ 7.63 × 10⁻¹⁴ |

The linear regression model achieves a near-perfect fit on the held-out test set. Because this dataset compiles *real* published rankings (World Happiness Report, HDI, GDP per capita, life expectancy, governance and environmental indices, etc.), an R² this close to 1.0 is unusually high and should not be taken at face value as evidence that happiness is trivially predictable from the other indicators. Two explanations are the most plausible:

1. **Methodological overlap between indices.** Several of the source indices used to construct `Happiness_Rank` (e.g., the World Happiness Report's own model) already incorporate GDP per capita, life expectancy, and perceived corruption as explicit input variables. If the compiled dataset preserves this overlap, the regression may effectively be "recovering" a known linear formula rather than discovering a new empirical relationship — a form of **data leakage** rather than genuine predictive insight.
2. **High shared variance across all rank indicators.** As the correlation heatmap shows, nearly all indicators are highly inter-correlated (ρ > 0.85 in most pairs), so a linear combination of ten strongly correlated predictors can reconstruct one more with very little residual error even without direct formula overlap.

This result should therefore be reported as a **pipeline sanity check rather than a generalizable finding**, and should be validated further — for example, by checking the exact index methodologies for shared inputs, by testing on more recent or held-out years only, or by dropping the indicators most likely to overlap methodologically with `Happiness_Rank` (`GDP_Per_Capita_Rank`, `Life_Expectancy_Rank`, `Corruption_Perception_Rank`) and re-evaluating model performance.

## Repository Structure

```
.
├── dataset/
│   └── global_country_rankings_2000_2026.csv
├── data analysis plots/          # Output of global_country_analysis.py
│   ├── correlation_heatmap_analysis.png
│   ├── Analysis_by_Regions.png
│   └── ...
├── ml plots/                     # Output of global_country_ml.py
│   └── plot_of_test_vs_predicted.png
├── global_country_analysis.py    # EDA pipeline
├── global_country_ml.py          # Regression pipeline
├── .gitignore
└── README.md
```

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

## Limitations and Future Work

- The near-perfect R² obtained by the linear model suggests strong (near-)deterministic structure among the ranking variables in this specific dataset; this should be explicitly validated against the original data-generating sources before drawing substantive conclusions about real-world predictability of happiness rankings.
- The analysis treats each country-year as an independent observation; no explicit time-series or panel-data structure (e.g., fixed effects, autocorrelation) is modeled.
- Future work could incorporate regularized regression (Ridge/Lasso) for feature-importance interpretation, non-linear models for benchmarking, and out-of-sample validation on external happiness-ranking datasets.

## License

Specify a license (e.g., MIT) here before publishing.
