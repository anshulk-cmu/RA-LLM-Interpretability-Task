# LLM Interpretability for Real Estate Valuation
## A Comparative Study of Reasoning Transparency Across Frontier Models

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Weave](https://img.shields.io/badge/Tracking-W%26B%20Weave-orange.svg)](https://wandb.ai)

- **Author:** Anshul Kumar
- **Email:** anshulk@andrew.cmu.edu
- **Institution:** Carnegie Mellon University
- **Project Type:** Research Assistant Application - LLM Interpretability Task
- **Date:** November 2025
- **Repository:** https://github.com/anshulk-cmu/RA-LLM-Interpretability-Task

---

## 🎯 Executive Summary

This project investigates **how Large Language Models reason about real estate valuation**, prioritizing **reasoning transparency and interpretability** over prediction accuracy. We treat valuation variance as a feature to understand model reasoning patterns rather than optimize for RMSE.

**Key Achievement:** Executed 540 API calls (36 properties × 5 prompting strategies × 3 models) with comprehensive observability, statistical analysis, and interpretability insights.

---

## 📊 Dataset & Market Dynamics

**36 comparable home sales** (1925-1965 construction era, all 3-bedroom properties):

| Feature | Min | Max | Mean | Std Dev |
|---------|-----|-----|------|---------|
| **Sale Price** | $128,000 | $283,000 | $203,581 | $38,245 |
| **Bathrooms** | 1 | 3 | 2.1 | 0.5 |
| **Lot Size (sq ft)** | 3,001 | 9,387 | 6,787 | 1,542 |
| **Year Built** | 1925 | 1965 | 1952 | 11.3 |

**Data Quality:** Zero missing values, 2 duplicates removed, normal price distribution (Shapiro-Wilk p > 0.05), no extreme outliers.

### Counterintuitive Correlations

| Feature | Correlation | Insight |
|---------|-------------|---------|
| **Price per sq ft (lot)** | **+0.66** | Land efficiency > absolute size |
| **Home Age** | **+0.34** | Older homes command premium |
| **Year Built** | **-0.34** | Pre-war > post-war value |
| **Lot Size** | **-0.31** | Urban density premium |
| **Bathrooms** | **-0.03** | Minimal impact |

**Hypothesis:** Urban historic district where pre-war character and walkability outweigh modern amenities.

---

## 🧪 Experimental Design

### Models Tested
- **Claude Sonnet 4.5** (Anthropic Direct API) - Advanced reasoning
- **Mistral Large 2411** (AWS Bedrock) - European training data
- **Llama 3.3 70B** (AWS Bedrock) - Open-source benchmark

**Parameters:** Temperature 0.3, Max Tokens 2000

### 5 Prompting Strategies

1. **Zero-Shot** - Raw valuation instinct, no context
2. **Few-Shot** - 3 comparable sales provided
3. **Chain-of-Thought** - 4-step reasoning decomposition (lot value → structure → depreciation → final)
4. **Role-Playing** - Certified appraiser persona (20 years experience)
5. **Constraint-Based** - Market context ($100K-$500K range, neighborhood characteristics)

### Execution
**Parallel processing:** 3 models simultaneously per strategy with 15s batch delays
**Runtime:** ~75 minutes | **Observability:** W&B Weave tracking (latency, tokens, costs)

---

## 📈 Results & Key Findings

### Overall Performance

| Metric | Value |
|--------|-------|
| **Total Predictions** | 540 |
| **Success Rate** | ~95% |
| **Mean Absolute % Error** | 15-25% (model/strategy dependent) |
| **Median Latency** | 2-8 seconds |
| **Token Usage** | 400-1200 tokens |

### Model Rankings (by Mean % Error)

| Rank | Model | Error | Latency | Tokens | Profile |
|------|-------|-------|---------|--------|---------|
| 1 | **Claude Sonnet 4.5** | Lowest | Medium | High | Best reasoning depth |
| 2 | **Mistral Large 2411** | Medium | Fastest | Medium | Speed-accuracy balance |
| 3 | **Llama 3.3 70B** | Highest | Slow | Low | Token-efficient |

**Statistical Significance:** ANOVA confirms significant differences (p < 0.05)

### Strategy Rankings (by Mean % Error)

| Rank | Strategy | Error | Variance | Coefficient of Variation |
|------|----------|-------|----------|--------------------------|
| 1 | **Few-Shot** | Lowest | Low | 0.12 (most consistent) |
| 2 | **Constraint-Based** | Low-Med | Low | 0.18 |
| 3 | **Chain-of-Thought** | Medium | Medium | 0.24 |
| 4 | **Zero-Shot** | High | High | 0.35 |
| 5 | **Role-Playing** | Highest | Highest | 0.42 (overconfident) |

### Best & Worst Combinations

**Top Performers:**
1. Claude + Few-Shot → 12% error
2. Mistral + Constraint-Based → 14% error
3. Claude + Constraint-Based → 15% error

**Worst Performers:**
1. Llama + Role-Playing → 35% error
2. Llama + Zero-Shot → 32% error
3. Mistral + Role-Playing → 28% error

---

## 🔍 Interpretability Insights

### Feature Discovery (Chain-of-Thought Analysis)
- ✅ **Claude:** Mentions "land efficiency" in 45% of responses (discovers key correlation)
- ⚠️ **Mistral:** Defaults to "larger lot = higher price" heuristic (misses pattern)
- ❌ **Llama:** Rarely mentions lot size explicitly

### Reasoning Structure
- **Claude:** Structured, follows 4-step template, numerical decomposition
- **Mistral:** Semi-structured, often skips depreciation step
- **Llama:** Narrative style, less quantitative

### Price Range Sensitivity

| Price Range | Mean % Error | Count | Pattern |
|-------------|--------------|-------|---------|
| <$150K | 22% | 45 | Underestimation |
| $150-200K | 15% | 180 | Best performance |
| $200-250K | 18% | 230 | Good performance |
| >$250K | 28% | 85 | Overestimation plateau |

**Key Finding:** Models excel at center of distribution, struggle with tails.

---

## 💰 Cost-Efficiency

**Claude Sonnet 4.5:**
- Mean cost: $0.0045/prediction (~0.45 cents)
- Total study cost: $2.43 (540 calls)
- Range: $0.0025 - $0.0080

**Efficiency Index (speed × accuracy):**
1. **Mistral** - Best overall balance
2. **Claude** - Accuracy > speed
3. **Llama** - Speed ≠ accuracy

---

## 🏗️ Technical Implementation

**Architecture:** 6-layer pipeline (Data Processing → Prompt Engineering → LLM Inference → Response Processing → Observability → Analysis)

**Feature Engineering:** 8 derived features (home_age, price_per_sqft_lot, lotsize_zscore, construction_era, has_extra_bath, age_category, price_quartile)

**Price Extraction:** Multi-pattern regex cascade with $50K-$1M validation, 95% success rate

**Parallel Execution:** ThreadPoolExecutor (3 workers), 3x speedup vs sequential

**Observability:** W&B Weave with @weave.op() decorators tracking latency, tokens, costs

**Statistical Analysis:** ANOVA, correlations, Cohen's d effect sizes, variance decomposition

---

## 📁 Repository Structure

```
├── ra_llm_interpretability_task.py    # Complete pipeline (2,200 lines)
├── RA_LLM_Interpretability_Task.ipynb # Jupyter notebook
├── README.md                           # This file
└── [Generated outputs: CSVs, JSONs, visualizations]
```

---

## 🔬 Research Questions Answered

1. **Inter-Model Agreement:** Models show significant variance (ANOVA p < 0.05)
2. **Prompt Sensitivity:** CV ranges 0.12 (few-shot) to 0.42 (role-playing)
3. **Feature Discovery:** Only Claude consistently discovers land efficiency pattern
4. **Reasoning Transparency:** Chain-of-Thought most interpretable, role-playing least reliable
5. **Cost-Accuracy Tradeoffs:** Claude costs 4.5× baseline, delivers lowest error
6. **Price Range Sensitivity:** 22-28% error at tails vs 15% at center
7. **Extraction Reliability:** 95% success, role-playing strategy most problematic

---

## 📧 Contact

**Anshul Kumar** | Carnegie Mellon University
Email: anshulk@andrew.cmu.edu | GitHub: [@anshulk-cmu](https://github.com/anshulk-cmu)

---

**License:** MIT | **Built at:** Carnegie Mellon University, November 2025
