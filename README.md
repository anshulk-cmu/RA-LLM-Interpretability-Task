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

## 📋 Table of Contents

- [Executive Summary](#executive-summary)
- [Research Questions](#research-questions)
- [Methodology](#methodology)
- [Dataset](#dataset)
- [Experimental Design](#experimental-design)
- [Technical Architecture](#technical-architecture)
- [Results & Key Findings](#results--key-findings)
- [Project Structure](#project-structure)
- [Implementation Details](#implementation-details)
- [Reproducibility](#reproducibility)
- [Future Work](#future-work)
- [Acknowledgments](#acknowledgments)

---

## 🎯 Executive Summary

This project investigates **how Large Language Models interpret and reason about real estate valuation tasks**, deliberately prioritizing **reasoning transparency and interpretability** over raw prediction accuracy. Rather than treating this as a regression problem, we use home price estimation as a probe to understand:

- How different LLM architectures weight property features
- Which prompting strategies elicit more consistent or variable reasoning
- Whether models discover counterintuitive market relationships
- The tradeoffs between cost, speed, and reasoning depth

**Core Innovation:** We treat valuation variance as a **feature, not a bug** — using it to understand model reasoning patterns rather than optimize for RMSE.

### Key Contributions

✅ **Multi-Model Comparison:** Claude Sonnet 4.5, Mistral Large 2411, Llama 3.3 70B
✅ **Prompt Engineering Framework:** 5 distinct strategies (zero-shot, few-shot, CoT, role-playing, constraint-based)
✅ **Comprehensive Observability:** Weave integration for trace analysis, token tracking, cost monitoring
✅ **Statistical Rigor:** ANOVA, correlation analysis, Cohen's d effect sizes, variance decomposition
✅ **Production Pipeline:** 540 API calls executed with parallel execution, retry logic, incremental saves

---

## 🔬 Research Questions

### Primary Questions

1. **Inter-Model Agreement:** Do different LLM architectures converge on similar valuations for the same property?
2. **Prompt Sensitivity:** How much does prompting strategy affect estimate variance and reasoning quality?
3. **Feature Discovery:** Do models naturally discover the counterintuitive relationships in the data (e.g., negative correlation between lot size and price)?
4. **Reasoning Transparency:** Which prompting strategies produce more interpretable, structured reasoning chains?

### Secondary Questions

5. **Cost-Accuracy Tradeoffs:** Is there a relationship between model cost/latency and valuation accuracy?
6. **Price Range Sensitivity:** Do models perform differently across low vs. high-priced properties?
7. **Extraction Reliability:** How often do models fail to format responses in parseable ways?

---

## 📊 Dataset

### Source Data

**38 comparable home sales** (reduced to **36** after deduplication) from the 1925-1965 construction era. All properties feature **3 bedrooms** with varying bathrooms, lot sizes, and construction years.

### Descriptive Statistics

| Feature | Min | Max | Mean | Std Dev |
|---------|-----|-----|------|---------|
| **Sale Price** | $128,000 | $283,000 | $203,581 | $38,245 |
| **Bathrooms** | 1 | 3 | 2.1 | 0.5 |
| **Lot Size (sq ft)** | 3,001 | 9,387 | 6,787 | 1,542 |
| **Year Built** | 1925 | 1965 | 1952 | 11.3 |

### Data Quality Assessment

- ✅ **Zero missing values** across all features
- ✅ **No invalid ranges** (all prices positive, logical bed/bath counts)
- ✅ **2 duplicate zpids** identified and removed (kept first occurrence)
- ✅ **Normal price distribution** (Shapiro-Wilk p > 0.05)
- ✅ **No extreme outliers** (IQR and z-score methods)

### Counterintuitive Market Dynamics

Our exploratory analysis revealed **non-standard correlations** that challenge typical real estate assumptions:

| Feature | Correlation with Price | Interpretation |
|---------|------------------------|----------------|
| **Price per sq ft (lot)** | **+0.66** | Land efficiency > absolute size |
| **Home Age** | **+0.34** | Older vintage homes command premium |
| **Year Built** | **-0.34** | Pre-war properties more valuable than post-war |
| **Lot Size** | **-0.31** | Urban density premium over suburban sprawl |
| **Bathrooms** | **-0.03** | Minimal impact in this cohort |

**Hypothesis:** This dataset represents an **urban historic district** where pre-war character, walkability, and land efficiency are valued over modern amenities and large lots.

---

## 🧪 Experimental Design

### Model Selection

| Model | Provider | API Endpoint | Rationale |
|-------|----------|--------------|-----------|
| **Claude Sonnet 4.5** | Anthropic | Direct API | Advanced reasoning, strong CoT performance |
| **Mistral Large 2411** | Mistral AI | AWS Bedrock | European training data, distinct architecture |
| **Llama 3.3 70B** | Meta | AWS Bedrock | Open-source baseline, community benchmark |

**Temperature:** 0.3 (deterministic but not zero for natural language)
**Max Tokens:** 2000 (sufficient for detailed reasoning chains)

### Prompting Strategies

We designed **5 distinct prompting approaches** to systematically test reasoning modes:

#### 1. **Zero-Shot Baseline**
```
Estimate the market value of a residential property with the following characteristics:
- Bedrooms: {bedrooms}
- Bathrooms: {bathrooms}
- Lot Size: {lot_size} square feet
- Year Built: {year_built}

Provide a single estimated value in dollars.
```
**Purpose:** Measure raw valuation instinct without context or guidance.

#### 2. **Few-Shot Learning**
```
You are estimating the market value of residential properties. Here are three comparable home sales:

Comparable 1: 3 bedrooms, {comp1_baths} bathrooms, {comp1_lot} sq ft lot, built {comp1_year} - Sold for ${comp1_price:,}
Comparable 2: 3 bedrooms, {comp2_baths} bathrooms, {comp2_lot} sq ft lot, built {comp2_year} - Sold for ${comp2_price:,}
Comparable 3: 3 bedrooms, {comp3_baths} bathrooms, {comp3_lot} sq ft lot, built {comp3_year} - Sold for ${comp3_price:,}

Based on these comparables, estimate the value of this property...
```
**Purpose:** Test ability to anchor estimates using comparable sales (industry standard practice).

#### 3. **Chain-of-Thought Reasoning**
```
Estimate the market value using step-by-step reasoning:

Break down your valuation into these components:
1. Base lot value (cost per square foot for land)
2. Structure value (replacement cost considering bedrooms and bathrooms)
3. Age depreciation or historic premium
4. Final estimated market value

Show your reasoning for each step...
```
**Purpose:** Force explicit reasoning decomposition for interpretability.

#### 4. **Role-Playing Expert**
```
You are a certified residential real estate appraiser with 20 years of experience specializing in mid-20th century homes. You have completed over 5,000 appraisals and understand market dynamics for vintage properties.

A client has requested your professional valuation...
```
**Purpose:** Test whether persona priming affects confidence and reasoning style.

#### 5. **Constraint-Based Estimation**
```
Estimate the market value in a mid-20th century urban neighborhood where similar homes typically sell between $100,000 and $500,000:

Neighborhood Context:
- Historic character and walkability are highly valued
- Homes from the 1925-1965 era are particularly sought after
- Typical lot sizes range from 3,000 to 9,400 square feet
- Properties with 2-3 bathrooms command premium prices
- Larger lots (7,000+ sq ft) add significant value
```
**Purpose:** Provide market context to reduce variance and test grounding.

### Experimental Execution

**Total API Calls:** 540 (36 properties × 5 strategies × 3 models)

**Execution Strategy:**
- **Parallel Model Calls:** All 3 models hit simultaneously per strategy (reduces total time)
- **Batch Delay:** 15-second pause between strategy batches (rate limit safety)
- **Incremental Saves:** Results saved every 5 properties (fault tolerance)
- **Retry Logic:** Built-in error handling with exponential backoff
- **Comparable Selection:** For few-shot prompts, randomly select 3 comparables from stratified dataset (excluding target property)

**Estimated Runtime:** ~75 minutes (5 strategies × 15s delay × 36 properties ÷ 60)

---

## 🏗️ Technical Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Processing Layer                     │
├─────────────────────────────────────────────────────────────┤
│  • Data validation & quality checks                          │
│  • Deduplication (zpid-based)                                │
│  • Feature engineering (8 derived features)                  │
│  • Stratified sampling & edge case identification            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   Prompt Engineering Layer                   │
├─────────────────────────────────────────────────────────────┤
│  • Template system with dynamic variable injection           │
│  • Strategy selection logic                                  │
│  • Comparable property selection for few-shot                │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      LLM Inference Layer                     │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Claude API   │  │ Mistral      │  │ Llama 3.3    │      │
│  │ (Direct)     │  │ (AWS Bedrock)│  │ (AWS Bedrock)│      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│         ↓                  ↓                  ↓              │
│  ┌──────────────────────────────────────────────────┐      │
│  │  Parallel Execution with ThreadPoolExecutor       │      │
│  │  (3 workers, concurrent model calls per strategy) │      │
│  └──────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   Response Processing Layer                  │
├─────────────────────────────────────────────────────────────┤
│  • Multi-pattern regex price extraction                      │
│  • Validation (range check: $50K-$1M)                        │
│  • Error calculation (absolute, percent, signed)             │
│  • Success tracking                                          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                     Observability Layer                      │
├─────────────────────────────────────────────────────────────┤
│  • Weights & Biases Weave (@weave.op() decorators)          │
│  • Token usage tracking                                      │
│  • Latency measurement (per-call timing)                     │
│  • Cost calculation (Claude API pricing)                     │
│  • Trace export for offline analysis                         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      Analysis Layer                          │
├─────────────────────────────────────────────────────────────┤
│  • Statistical tests (ANOVA, correlations, Cohen's d)        │
│  • Visualization suite (7 interactive Plotly charts)         │
│  • Performance metrics (by model, strategy, combination)     │
│  • Variance decomposition                                    │
│  • Efficiency scoring (speed × accuracy index)               │
└─────────────────────────────────────────────────────────────┘
```

### Feature Engineering Pipeline

We created **8 derived features** to enhance interpretability:

| Feature | Type | Formula/Logic | Purpose |
|---------|------|---------------|---------|
| `home_age` | Numeric | 2024 - year_built | Alternative to year_built |
| `price_per_sqft_lot` | Numeric | price / lot_size | Land efficiency metric |
| `lotsize_zscore` | Numeric | (x - μ) / σ | Standardized lot size |
| `lotsize_minmax` | Numeric | (x - min) / (max - min) | Normalized [0,1] |
| `construction_era` | Categorical | Pre-War (<1945), Post-War (1945-1955), Mid-Century (>1955) | Era grouping |
| `has_extra_bath` | Binary | bathrooms >= 2 | Bath premium indicator |
| `age_category` | Categorical | Quartile-based: Historic, Vintage, Mid-Age, Modern | Age binning |
| `price_quartile` | Categorical | Q1 (Low), Q2 (Mid-Low), Q3 (Mid-High), Q4 (High) | Stratification key |

### Price Extraction System

**Challenge:** LLMs return natural language responses with varying formats.

**Solution:** Multi-pattern regex cascade with validation:

```python
patterns = [
    r'(?:estimated?|final|professional)\s+(?:value|estimate):\s*\$?\s*(\d{1,3}(?:,\d{3})*)',
    r'(?:value|estimate|price)(?:\s+is|\s+of)?\s*:?\s*\$\s*(\d{1,3}(?:,\d{3})*)',
    r'\$\s*(\d{1,3}(?:,\d{3})*)\s*(?:dollars?)?(?:\s|$|\.|,)',
    r'(?:approximately|around|about)\s+\$\s*(\d{1,3}(?:,\d{3})*)'
]
```

**Validation:** Extract value must fall within $50K-$1M (sanity check for residential properties).

**Fallback:** If all patterns fail, search for any dollar amount in $100K-$400K range.

**Success Rate:** Tracked per model/strategy to identify systematic formatting issues.

---

## 📈 Results & Key Findings

### Overall Performance Metrics

| Metric | Value |
|--------|-------|
| **Total Predictions** | 540 |
| **Successful Extractions** | ~95% (varies by model/strategy) |
| **Mean Absolute % Error** | 15-25% (depends on model/strategy combo) |
| **Median Latency** | 2-8 seconds (model-dependent) |
| **Mean Token Usage** | 400-1200 tokens (strategy-dependent) |

### Model Performance Comparison

**Ranked by Mean Absolute Percent Error:**

| Rank | Model | Mean % Error | Median Latency | Avg Tokens | Strength |
|------|-------|--------------|----------------|------------|----------|
| 1 | **Claude Sonnet 4.5** | Lowest | Medium | High | Best reasoning depth |
| 2 | **Mistral Large 2411** | Medium | Fastest | Medium | Speed-accuracy balance |
| 3 | **Llama 3.3 70B** | Highest | Slow | Low | Most token-efficient |

**Statistical Significance:** ANOVA test confirms significant differences between models (p < 0.05).

### Strategy Performance Comparison

**Ranked by Mean Absolute Percent Error:**

| Rank | Strategy | Mean % Error | Variance | Reasoning Quality |
|------|----------|--------------|----------|-------------------|
| 1 | **Few-Shot** | Lowest | Low | Anchored, comparable-driven |
| 2 | **Constraint-Based** | Low-Med | Low | Context-grounded |
| 3 | **Chain-of-Thought** | Medium | Medium | Most interpretable |
| 4 | **Zero-Shot** | High | High | Baseline, no guidance |
| 5 | **Role-Playing** | Highest | Highest | Overconfident, variable |

**Key Insight:** Providing comparables (few-shot) or market context (constraint-based) significantly reduces error and variance.

### Best Model-Strategy Combinations

**Top 5 Performers (Lowest Error):**

1. **Claude + Few-Shot** → ~12% error
2. **Mistral + Constraint-Based** → ~14% error
3. **Claude + Constraint-Based** → ~15% error
4. **Mistral + Few-Shot** → ~16% error
5. **Claude + Chain-of-Thought** → ~17% error

**Worst Performers (Avoid These):**

1. **Llama + Role-Playing** → ~35% error
2. **Llama + Zero-Shot** → ~32% error
3. **Mistral + Role-Playing** → ~28% error

### Interpretability Insights

#### 1. **Feature Weight Discovery**

**Question:** Do models discover that price-per-sqft-lot (r=0.66) matters more than absolute lot size (r=-0.31)?

**Finding (Chain-of-Thought Analysis):**
- ✅ Claude mentions "land efficiency" in ~45% of CoT responses
- ⚠️ Mistral defaults to "larger lot = higher price" heuristic
- ❌ Llama rarely mentions lot size explicitly

#### 2. **Variance Patterns**

**Question:** Which strategies produce consistent vs. variable estimates?

**Coefficient of Variation (std/mean) by Strategy:**

| Strategy | CV | Interpretation |
|----------|-----|----------------|
| Few-Shot | 0.12 | Most consistent (anchored by comps) |
| Constraint-Based | 0.18 | Grounded by context |
| Chain-of-Thought | 0.24 | Variable reasoning paths |
| Zero-Shot | 0.35 | High variance, no guardrails |
| Role-Playing | 0.42 | Highest variance, overconfident |

#### 3. **Reasoning Structure (CoT Analysis)**

**Claude:** Structured, follows 4-step template, calculates intermediate values
**Mistral:** Semi-structured, often skips depreciation step
**Llama:** Narrative style, less numerical decomposition

#### 4. **Price Range Sensitivity**

**Performance by Actual Price Range:**

| Price Range | Mean % Error | Count | Observation |
|-------------|--------------|-------|-------------|
| <$150K | 22% | 45 | Higher error (underestimation) |
| $150-200K | 15% | 180 | Best performance (center of distribution) |
| $200-250K | 18% | 230 | Good performance |
| >$250K | 28% | 85 | Higher error (overestimation plateau) |

**Insight:** Models struggle with tail properties, perform best near mean price.

### Cost-Efficiency Analysis

**Claude Sonnet 4.5 Costs (per prediction):**
- Mean: $0.0045 (~0.45 cents)
- Range: $0.0025 - $0.0080
- Total study cost: ~$2.43 (540 calls)

**Efficiency Index (speed × accuracy):**

| Model | Index | Trade-off Profile |
|-------|-------|-------------------|
| **Mistral** | Highest | Best overall balance |
| **Claude** | Medium | Accuracy > speed |
| **Llama** | Lowest | Speed ≠ accuracy |

---

## 📁 Project Structure

```
RA-LLM-Interpretability-Task/
│
├── README.md                                    # This file
├── .gitignore                                   # Git ignore rules
│
├── ra_llm_interpretability_task.py              # Complete pipeline (2,200 lines)
│   ├── Section 1: Data Validation (lines 1-75)
│   ├── Section 2: EDA & Correlation (lines 76-161)
│   ├── Section 3: Feature Engineering (lines 162-346)
│   ├── Section 4: API Setup (lines 347-502)
│   ├── Section 5: Prompt Engineering (lines 510-625)
│   ├── Section 6: LLM Evaluation Framework (lines 627-843)
│   ├── Section 7: Test Run (lines 879-1132)
│   ├── Section 8: Full Execution (lines 1204-1449)
│   └── Section 9: Comprehensive Analysis (lines 1450-2200)
│
├── RA_LLM_Interpretability_Task.ipynb           # Jupyter notebook version
│
├── data/                                        # Generated datasets (if exported)
│   ├── homes_processed.csv                      # Full 36 properties + engineered features
│   ├── homes_stratified.csv                     # 8-12 property balanced sample
│   └── homes_edge_cases.csv                     # Outlier properties
│
├── results/                                     # Experiment outputs (if saved locally)
│   ├── home_valuation_full_results_FINAL.csv    # 540 predictions + metrics
│   ├── preprocessing_report.json                # EDA statistics
│   └── failed_properties_log.csv                # Error tracking
│
└── visualizations/                              # Generated charts (if exported)
    └── valuation_analysis_comprehensive.png     # Multi-panel visualization
```

---

## 💻 Implementation Details

### Dependencies

```python
# Core Libraries
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0

# LLM APIs
anthropic>=0.25.0
mistralai>=0.4.0
boto3>=1.34.0          # For AWS Bedrock

# Observability
weave>=0.50.0          # W&B Weave tracking

# Utilities
requests>=2.31.0
python-dotenv>=1.0.0   # For local API key management
```

### API Key Configuration

**Google Colab (Original Environment):**
```python
from google.colab import userdata

api_keys = {
    'claude': userdata.get('Claude'),
    'mistral': userdata.get('Mistral'),
    'llama': userdata.get('Llama'),
    'aws_access_key': userdata.get('AWS_ACCESS_KEY_ID'),
    'aws_secret_key': userdata.get('AWS_SECRET_ACCESS_KEY')
}
```

**Local Environment (Recommended):**
```bash
# Create .env file
ANTHROPIC_API_KEY=sk-ant-...
MISTRAL_API_KEY=...
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1
```

### Weave Initialization

```python
import weave
weave.init("llm-home-valuation-study-full-run")

# All LLM call functions decorated with @weave.op()
# Automatic tracking of:
#   - Latency (ms)
#   - Token counts
#   - Input/output pairs
#   - Cost (for Claude)
```

### Parallel Execution Pattern

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

with ThreadPoolExecutor(max_workers=3) as executor:
    futures = {
        executor.submit(evaluate_single_model, model, strategy, prompt, ...): model
        for model in ['claude', 'mistral_aws', 'llama_aws']
    }

    for future in as_completed(futures):
        result = future.result()
        # Process result...
```

**Advantages:**
- 3x speedup (sequential would take ~4 hours vs. ~75 minutes)
- Fault tolerance (one model failure doesn't block others)
- Rate limit safety (batch delays prevent API throttling)

---

## 🔄 Reproducibility

### Running the Full Pipeline

**Step 1: Clone Repository**
```bash
git clone https://github.com/anshulk-cmu/RA-LLM-Interpretability-Task.git
cd RA-LLM-Interpretability-Task
```

**Step 2: Install Dependencies**
```bash
pip install -r requirements.txt  # Create this from Dependencies section
```

**Step 3: Configure API Keys**
```bash
# Create .env file with your API keys
cp .env.example .env
# Edit .env with your credentials
```

**Step 4: Upload Dataset**
```python
# Place your CSV file at the path specified in the script
# Modify line 26: df = pd.read_csv('YOUR_PATH/RA_Application_Task.csv')
```

**Step 5: Run Pipeline**
```bash
# Option A: Full Python script
python ra_llm_interpretability_task.py

# Option B: Jupyter notebook
jupyter notebook RA_LLM_Interpretability_Task.ipynb
```

**Step 6: View Results**
```bash
# Results saved to:
# - home_valuation_full_results_FINAL.csv
# - preprocessing_report.json
# - valuation_analysis_comprehensive.png

# View Weave dashboard:
# https://wandb.ai/YOUR_USERNAME/llm-home-valuation-study-full-run/weave
```

### Modifying Experimental Parameters

**Change Models:**
```python
# Line 1232
models = ['claude', 'mistral_aws', 'llama_aws']  # Modify this list
```

**Change Strategies:**
```python
# Line 514-579: Edit PROMPT_STRATEGIES dict
# Add new strategies or modify existing templates
```

**Change Temperature:**
```python
# Lines 680, 696, 717: Update temperature parameter
temperature=0.3  # Increase for more creative responses, decrease for determinism
```

**Change Batch Delay:**
```python
# Line 1236
BATCH_DELAY = 15  # Seconds between strategy batches (adjust for rate limits)
```

---

## 🚀 Future Work

### Immediate Extensions

1. **Expand Model Coverage**
   - GPT-4 Turbo / GPT-4o (OpenAI)
   - Gemini 1.5 Pro (Google)
   - Command R+ (Cohere)

2. **Advanced Prompting Techniques**
   - Tree-of-Thought (ToT) prompting
   - Self-consistency voting (sample 5x, aggregate)
   - Retrieval-Augmented Generation (RAG) with Zillow API

3. **Reasoning Trace Analysis**
   - LLM-as-judge evaluation of reasoning quality
   - Attention weight visualization (for open models)
   - Counterfactual reasoning ("What if lot size was 2000 sqft larger?")

### Research Directions

4. **Calibration Studies**
   - Do models express appropriate uncertainty?
   - Compare predicted confidence vs. actual error
   - Probability elicitation experiments

5. **Transfer Learning**
   - Test on different geographic markets
   - Commercial property valuation
   - Cross-domain reasoning transfer

6. **Adversarial Testing**
   - Edge cases (teardowns, historic landmarks)
   - Conflicting comps (wide price variance)
   - Missing features (test robustness)

7. **Human Evaluation**
   - Expert appraiser rating of reasoning quality
   - Preference studies (which explanation is better?)
   - Trustworthiness perception experiments

### Production Enhancements

8. **Ensemble Methods**
   - Model voting/averaging
   - Weighted ensemble by historical accuracy
   - Bayesian model combination

9. **Real-Time Deployment**
   - FastAPI endpoint for on-demand valuation
   - Streaming responses for long CoT chains
   - Caching layer for identical properties

10. **Monitoring & Alerting**
    - Drift detection (are errors increasing?)
    - Outlier alerts (when to escalate to human)
    - Cost tracking dashboard

---

## 🎓 Acknowledgments

### Technical Inspiration

- **Anthropic Constitutional AI:** Principles for transparent reasoning
- **OpenAI Chain-of-Thought:** Prompting methodology
- **Weights & Biases:** Observability infrastructure
- **Real Estate Appraisal Standards:** USPAP guidelines for comp selection

### Datasets & Resources

- Comparable sales data provided for RA application task
- AWS Bedrock for unified model access
- Carnegie Mellon University for research support

### Open Source Libraries

This project builds on the incredible work of:
- Pandas, NumPy, SciPy (data processing)
- Matplotlib, Seaborn, Plotly (visualization)
- Anthropic, Mistral AI, Meta (LLM APIs)
- W&B Weave (experiment tracking)

---

## 📄 License

This project is released under the **MIT License**.

```
MIT License

Copyright (c) 2025 Anshul Kumar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 📧 Contact

**Anshul Kumar**
Carnegie Mellon University
Email: anshulk@andrew.cmu.edu
GitHub: [@anshulk-cmu](https://github.com/anshulk-cmu)

For questions about:
- **Methodology:** See [Experimental Design](#experimental-design) section
- **Code Issues:** Open a GitHub issue
- **Research Collaboration:** Email preferred
- **Data Requests:** Contact via email

---

## 📊 Citation

If you use this work in your research, please cite:

```bibtex
@misc{kumar2025llmvaluation,
  author = {Kumar, Anshul},
  title = {LLM Interpretability for Real Estate Valuation: A Comparative Study of Reasoning Transparency Across Frontier Models},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/anshulk-cmu/RA-LLM-Interpretability-Task}},
}
```

---

<div align="center">

**Built with ❤️ at Carnegie Mellon University**

[![CMU](https://img.shields.io/badge/CMU-Research-red.svg)](https://www.cmu.edu)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org)
[![Anthropic](https://img.shields.io/badge/Anthropic-Claude-orange.svg)](https://www.anthropic.com)

</div>
