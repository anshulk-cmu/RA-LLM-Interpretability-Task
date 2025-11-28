# LLM Interpretability for Real Estate Valuation

- **Author:** Anshul Kumar
- **Email:** anshulk@andrew.cmu.edu
- **Project:** Research Assistant Application - LLM Interpretability Task
- **Date:** November 2025

## Project Overview

This task investigates how Large Language Models interpret and reason about real estate valuation tasks. Rather than optimizing for prediction accuracy, the task focuses on understanding the variance, reasoning transparency, and interpretability of AI-generated home price estimates across multiple models and prompting strategies.

## Dataset Description

The dataset comprises 38 comparable home sales from the 1925-1965 construction era, reduced to 36 properties after deduplication. All homes feature 3 bedrooms with the following characteristics:

- **Price Range:** $128,000 - $283,000 (Mean: $203,581)
- **Bathrooms:** 1-3 (predominantly 2 bathrooms)
- **Lot Size:** 3,001 - 9,387 sq ft (Mean: 6,787 sq ft)
- **Year Built:** 1925-1965 (Mean: 1952)

## Completed Work

### 1. Data Preprocessing & Quality Assessment
Performed comprehensive data validation revealing zero missing values, no invalid ranges, and identified 2 duplicate property IDs. Quality checks confirmed all prices are positive, bedroom/bathroom counts are logical, and construction years fall within the expected historic range.

### 2. Exploratory Data Analysis
Statistical analysis uncovered counterintuitive market dynamics. Contrary to typical assumptions, newer homes within this vintage range show negative correlation with price (r = -0.34), suggesting premium valuation for pre-war properties. Similarly, larger lot sizes correlate negatively with price (r = -0.31), indicating urban density premium over suburban sprawl.

The Shapiro-Wilk normality test confirms sale prices follow a normal distribution (p > 0.05), with slight positive skewness (0.22) indicating some higher-priced outliers. No extreme outliers were detected using IQR and z-score methods.

### 3. Feature Engineering
Created 8 derived features to enhance model interpretability:
- **home_age**: Current age from 2024 baseline
- **price_per_sqft_lot**: Land efficiency metric (strongest correlation at 0.66)
- **lotsize_zscore & lotsize_minmax**: Normalized lot size representations
- **construction_era**: Categorical grouping (Pre-War, Post-War, Mid-Century)
- **has_extra_bath**: Binary indicator for 2+ bathrooms
- **age_category**: Quartile-based binning (Historic, Vintage, Mid-Age, Modern)
- **price_quartile**: Stratified sampling framework

### 4. Multi-Model LLM Infrastructure
Successfully established API connections and verified functionality for three frontier models:
- **Claude Sonnet 4.5** (Anthropic): Advanced reasoning capabilities
- **Mistral Large 2411**: European alternative with distinct training
- **Llama 3.3 70B** (via OpenRouter): Open-source benchmark

### 5. Observability Stack
Integrated three complementary monitoring platforms:
- **Weights & Biases Weave**: Experiment tracking and MLOps
- **Arize Phoenix**: Real-time LLM trace analysis
- **Langfuse**: Prompt versioning and cost analytics

### 6. Data Export
Generated three stratified datasets for systematic experimentation:
- `homes_processed.csv`: Full 36-property dataset with engineered features
- `homes_stratified.csv`: Balanced 8-property sample across price quartiles
- `homes_edge_cases.csv`: Outlier properties for stress testing

## Key Findings

The strongest predictor of price is **price-per-square-foot of lot** (r = 0.66), revealing land efficiency matters more than absolute size. Home age shows moderate positive correlation (r = 0.34), while bathroom count has minimal impact (r = -0.03). These insights will guide prompt design to test whether LLMs naturally discover these relationships or require explicit guidance.

## Next Steps

### Phase 1: Prompt Strategy Implementation
Design and execute 5 distinct prompting approaches across all 3 models (15 experimental conditions):
1. Zero-shot baseline valuation
2. Few-shot with 3 comparable examples
3. Chain-of-thought reasoning decomposition
4. Role-playing as certified appraiser
5. Constraint-based estimation with bounds

### Phase 2: Systematic Valuation Experiment
Run 540 total API calls (36 homes × 15 conditions) to generate comprehensive valuation estimates. Capture mean, standard deviation, min/max, and confidence intervals per property per condition.

### Phase 3: Comparative Analysis
Statistical framework examining:
- Inter-model agreement and correlation patterns
- Prompt sensitivity analysis (variance reduction techniques)
- Error decomposition by property features
- Cost-accuracy-interpretability tradeoffs
- Reasoning chain structure analysis via observability traces

### Phase 4: Final Deliverables
- Executive summary with novel insights beyond accuracy metrics
- Interactive visualization dashboard
- Production-ready reproducible pipeline
- Research methodology documentation

## Results

This work demonstrates that LLM valuation tasks offer a unique lens into model interpretability. By deliberately de-emphasizing accuracy and focusing on variance and reasoning transparency, we can understand which features models naturally weight, how prompting shapes reasoning chains, and where different architectures exhibit systematic biases.

---

- **Repository:** https://github.com/anshulk-cmu/RA-LLM-Interpretability-Task
- **Submission:** RA - LLM Interpretability Task
