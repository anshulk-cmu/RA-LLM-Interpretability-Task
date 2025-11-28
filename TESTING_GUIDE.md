# LLM Valuation Testing Guide

## Overview

This guide explains how to run the single-home valuation test across all 3 models and 5 prompting strategies.

## Test Configuration

**Test Home:**
- Property ID: 11469598
- Bedrooms: 3
- Bathrooms: 2
- Lot Size: 6,695 sq ft
- Year Built: 1958
- **Actual Sale Price: $207,000**

**Models Tested:**
1. Claude Sonnet 4.5 (Anthropic)
2. Mistral Large 2411
3. Llama 3.3 70B (via OpenRouter)

**Prompting Strategies:**
1. **Zero-Shot**: Basic valuation request with minimal context
2. **Few-Shot**: Provides 3 comparable sales as examples
3. **Chain-of-Thought**: Requires step-by-step reasoning breakdown
4. **Role-Playing**: Sets up expert appraiser persona
5. **Constraint-Based**: Includes neighborhood context and price bounds

**Total API Calls:** 15 (5 strategies × 3 models)

## Setup Instructions

### 1. Install Dependencies

For local Python environment:
```bash
pip install anthropic mistralai pandas requests
```

For Google Colab (already installed in notebook):
```python
!pip install anthropic mistralai
```

### 2. Configure API Keys

#### Local Environment:
```bash
export ANTHROPIC_API_KEY='your_claude_key_here'
export MISTRAL_API_KEY='your_mistral_key_here'
export OPENROUTER_API_KEY='your_openrouter_key_here'
```

#### Google Colab:
Store in Colab Secrets with keys:
- `Claude` or `ANTHROPIC_API_KEY`
- `Mistral` or `MISTRAL_API_KEY`
- `Llama` or `OPENROUTER_API_KEY`

## Running the Test

### Option 1: Using Test Runner (Recommended)

```bash
python test_runner.py
```

This will:
- Check API key configuration
- Run all 15 experimental conditions
- Display results in a comparison table
- Save results to CSV with timestamp
- Generate summary statistics

### Option 2: Using Main Pipeline Directly

```bash
python llm_valuation_pipeline.py
```

### Option 3: In Google Colab

```python
# Import the pipeline
from llm_valuation_pipeline import (
    run_single_home_test,
    create_comparison_table,
    analyze_results
)

# Setup
test_home = {
    'zpid': '11469598',
    'bedrooms': 3,
    'bathrooms': 2,
    'lot_size': 6695,
    'year_built': 1958
}

actual_price = 207000

comparables = [
    {'bathrooms': 2, 'lot_size': 6000, 'year_built': 1957, 'price': 205000},
    {'bathrooms': 2, 'lot_size': 8450, 'year_built': 1958, 'price': 215000},
    {'bathrooms': 2, 'lot_size': 8925, 'year_built': 1960, 'price': 230000}
]

# Load API keys from Colab secrets
from google.colab import userdata
api_keys = {
    'claude': userdata.get('ANTHROPIC_API_KEY'),
    'mistral': userdata.get('MISTRAL_API_KEY'),
    'llama': userdata.get('Llama')
}

# Run test
results_df = run_single_home_test(test_home, actual_price, comparables, api_keys)

# Display comparison
comparison = create_comparison_table(results_df)
print(comparison)

# Analyze
analysis = analyze_results(results_df)
print(analysis)
```

## Expected Output

### 1. Comparison Table
```
Strategy          | claude    | mistral   | llama     | actual
------------------|-----------|-----------|-----------|--------
zero_shot         | $XXX,XXX  | $XXX,XXX  | $XXX,XXX  | $207,000
few_shot          | $XXX,XXX  | $XXX,XXX  | $XXX,XXX  | $207,000
chain_of_thought  | $XXX,XXX  | $XXX,XXX  | $XXX,XXX  | $207,000
role_playing      | $XXX,XXX  | $XXX,XXX  | $XXX,XXX  | $207,000
constraint_based  | $XXX,XXX  | $XXX,XXX  | $XXX,XXX  | $207,000
```

### 2. Summary Statistics
- Mean estimate across all models/strategies
- Standard deviation (variance measure)
- Min/max range
- Mean absolute error vs actual price
- Mean percent error
- Extraction success rate

### 3. Saved Files
- `test_results_YYYYMMDD_HHMMSS.csv`: Full results with all metadata
- `test_analysis_YYYYMMDD_HHMMSS.json`: Statistical analysis

## Results Analysis

### Key Questions to Answer:

1. **Model Agreement**: Do all three models produce similar estimates?
2. **Prompt Sensitivity**: Which strategy produces lowest variance?
3. **Accuracy**: Which model/strategy combination is closest to $207K?
4. **Reasoning Quality**: Does chain-of-thought provide interpretable breakdowns?
5. **Constraint Adherence**: Does constraint-based stay within $130K-$280K bounds?

### Analysis Metrics:

**By Model:**
- Mean estimate per model
- Standard deviation per model (consistency)
- Mean absolute error per model

**By Strategy:**
- Mean estimate per strategy
- Variance across models for each strategy
- Extraction success rate per strategy

## Troubleshooting

### API Key Errors
```
❌ AuthenticationError: Invalid API key
```
**Solution**: Double-check API key format and permissions

### Rate Limiting
```
❌ RateLimitError: Too many requests
```
**Solution**: Script includes 1-second delay between calls. If needed, increase in `run_single_home_test()`

### Value Extraction Failures
```
⚠️ Failed to extract value
```
**Solution**: Check `extract_price_from_text()` regex patterns. LLM response format may vary.

### Network Errors
```
❌ RequestError: Connection timeout
```
**Solution**: Script includes 3 retries with exponential backoff. Check network connection.

## Next Steps After Test

1. **Debug Issues**: Fix any extraction failures or API errors
2. **Tune Parameters**: Adjust temperature, max_tokens if needed
3. **Refine Prompts**: Improve prompts that produced poor extractions
4. **Scale to Full Batch**: Run on all 36 homes once validated
5. **Add Observability**: Integrate Weave/Phoenix/Langfuse tracking

## Cost Estimates

Approximate API costs for single-home test (15 calls):

| Model | Cost per 1K Tokens | Estimated Cost |
|-------|-------------------|----------------|
| Claude Sonnet 4.5 | $0.003 input / $0.015 output | ~$0.10-0.15 |
| Mistral Large | $0.002 input / $0.006 output | ~$0.05-0.08 |
| Llama 3.3 (free tier) | $0.00 | $0.00 |

**Total estimated cost: ~$0.20-0.25 per home**

For full dataset (36 homes × 15 calls = 540 API calls):
**Estimated total: ~$7-9**

## File Structure

```
RA-LLM-Interpretability-Task/
├── llm_valuation_pipeline.py    # Main pipeline with all functions
├── test_runner.py                # Simple test execution script
├── TESTING_GUIDE.md              # This file
├── test_results_*.csv            # Generated test results
└── test_analysis_*.json          # Generated analysis
```

## Support

For issues or questions:
- **Author**: Anshul Kumar
- **Email**: anshulk@andrew.cmu.edu
- **Repository**: https://github.com/anshulk-cmu/RA-LLM-Interpretability-Task
