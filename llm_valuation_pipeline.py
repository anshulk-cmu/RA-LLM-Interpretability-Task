"""
LLM Home Valuation Pipeline
Author: Anshul Kumar (anshulk@andrew.cmu.edu)
Purpose: Test multiple LLMs with various prompting strategies for home price estimation
"""

import os
import json
import time
import re
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PROMPT STRATEGY MATRIX
# ============================================================================

PROMPT_STRATEGIES = {

    # Strategy 1: Zero-Shot Baseline
    "zero_shot": """Estimate the market value of a residential property with the following characteristics:
- Bedrooms: {bedrooms}
- Bathrooms: {bathrooms}
- Lot Size: {lot_size} square feet
- Year Built: {year_built}

Provide your estimated value in dollars.""",

    # Strategy 2: Few-Shot with Comparables
    "few_shot": """You are estimating the market value of residential properties. Here are three comparable home sales:

Comparable 1: 3 bedrooms, {comp1_baths} bathrooms, {comp1_lot} sq ft lot, built {comp1_year} - Sold for ${comp1_price:,}
Comparable 2: 3 bedrooms, {comp2_baths} bathrooms, {comp2_lot} sq ft lot, built {comp2_year} - Sold for ${comp2_price:,}
Comparable 3: 3 bedrooms, {comp3_baths} bathrooms, {comp3_lot} sq ft lot, built {comp3_year} - Sold for ${comp3_price:,}

Based on these comparables, estimate the value of this property:
- Bedrooms: {bedrooms}
- Bathrooms: {bathrooms}
- Lot Size: {lot_size} square feet
- Year Built: {year_built}

Provide your estimated value in dollars.""",

    # Strategy 3: Chain-of-Thought Reasoning
    "chain_of_thought": """Estimate the market value of this residential property using step-by-step reasoning:

Property Details:
- Bedrooms: {bedrooms}
- Bathrooms: {bathrooms}
- Lot Size: {lot_size} square feet
- Year Built: {year_built}

Break down your valuation into these components:
1. Base lot value (considering size and location utility)
2. Structure value (accounting for age, bedrooms, bathrooms)
3. Depreciation or appreciation factors (historic value, condition assumptions)
4. Final estimated market value

Show your reasoning for each step, then provide your final estimate in dollars.""",

    # Strategy 4: Role-Playing Expert Appraiser
    "role_playing": """You are a certified real estate appraiser with 20 years of experience in residential property valuation. You specialize in mid-20th century homes and understand how factors like lot size, age, and historic character affect property values.

A client has asked you to provide a professional valuation estimate for this property:
- Bedrooms: {bedrooms}
- Bathrooms: {bathrooms}
- Lot Size: {lot_size} square feet
- Year Built: {year_built}

As an expert appraiser, provide your estimated market value in dollars, considering all relevant factors that would influence the value of a home of this vintage.""",

    # Strategy 5: Constraint-Based with Neighborhood Context
    "constraint_based": """Estimate the market value of this residential property in a mid-20th century neighborhood where similar homes typically sell between $130,000 and $280,000:

Property Details:
- Bedrooms: {bedrooms}
- Bathrooms: {bathrooms}
- Lot Size: {lot_size} square feet
- Year Built: {year_built}

Context: This neighborhood values historic character, walkability, and efficient use of urban land. Homes from the 1925-1965 era are particularly sought after. Lot sizes typically range from 3,000 to 9,400 square feet.

Based on these neighborhood characteristics and the property details, provide your estimated value in dollars."""
}


def format_prompt(strategy_name: str, home_data: Dict, comparables_data: Optional[List[Dict]] = None) -> str:
    """
    Format a prompt template with actual home data

    Args:
        strategy_name: One of the 5 strategy keys
        home_data: Dict with keys: bedrooms, bathrooms, lot_size, year_built
        comparables_data: List of 3 dicts (only needed for few_shot strategy)

    Returns:
        Formatted prompt string
    """
    template = PROMPT_STRATEGIES[strategy_name]

    if strategy_name == "few_shot":
        if not comparables_data or len(comparables_data) != 3:
            raise ValueError("few_shot strategy requires exactly 3 comparables")

        # Merge home data with comparables
        prompt_data = {
            **home_data,
            'comp1_baths': comparables_data[0]['bathrooms'],
            'comp1_lot': comparables_data[0]['lot_size'],
            'comp1_year': comparables_data[0]['year_built'],
            'comp1_price': comparables_data[0]['price'],
            'comp2_baths': comparables_data[1]['bathrooms'],
            'comp2_lot': comparables_data[1]['lot_size'],
            'comp2_year': comparables_data[1]['year_built'],
            'comp2_price': comparables_data[1]['price'],
            'comp3_baths': comparables_data[2]['bathrooms'],
            'comp3_lot': comparables_data[2]['lot_size'],
            'comp3_year': comparables_data[2]['year_built'],
            'comp3_price': comparables_data[2]['price'],
        }
        return template.format(**prompt_data)
    else:
        return template.format(**home_data)


# ============================================================================
# VALUE EXTRACTION
# ============================================================================

def extract_price_from_text(text: str) -> Optional[int]:
    """
    Extract dollar amount from LLM response text using multiple patterns

    Args:
        text: Response text from LLM

    Returns:
        Extracted price as integer, or None if not found
    """
    # Pattern 1: $XXX,XXX or $XXX XXX (with commas or spaces)
    pattern1 = r'\$\s*(\d{1,3}(?:[,\s]\d{3})*(?:\.\d{2})?)'

    # Pattern 2: XXX,XXX dollars or XXX XXX dollars
    pattern2 = r'(\d{1,3}(?:[,\s]\d{3})*)\s*dollars'

    # Pattern 3: Final estimate/value is XXX,XXX
    pattern3 = r'(?:estimate|value|worth)(?:\s+is)?\s*:?\s*\$?\s*(\d{1,3}(?:[,\s]\d{3})*)'

    patterns = [pattern1, pattern2, pattern3]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            # Take the last match (often the final estimate)
            price_str = matches[-1].replace(',', '').replace(' ', '').replace('$', '')
            try:
                return int(float(price_str))
            except ValueError:
                continue

    return None


# ============================================================================
# UNIFIED LLM CALLER
# ============================================================================

def call_llm(
    model_name: str,
    prompt: str,
    home_id: str,
    strategy_name: str,
    api_keys: Dict[str, str],
    max_retries: int = 3
) -> Dict[str, Any]:
    """
    Unified LLM caller with observability and error handling

    Args:
        model_name: One of ['claude', 'mistral', 'llama']
        prompt: Formatted prompt string
        home_id: Identifier for the home being valued
        strategy_name: Name of the prompting strategy
        api_keys: Dictionary of API keys
        max_retries: Number of retry attempts on failure

    Returns:
        Dictionary with response data and metadata
    """
    start_time = time.time()

    result = {
        'model': model_name,
        'prompt': prompt,
        'response': None,
        'estimated_value': None,
        'tokens': None,
        'latency': None,
        'strategy': strategy_name,
        'home_id': home_id,
        'timestamp': datetime.now().isoformat(),
        'error': None
    }

    for attempt in range(max_retries):
        try:
            if model_name == 'claude':
                result.update(_call_claude(prompt, api_keys))
            elif model_name == 'mistral':
                result.update(_call_mistral(prompt, api_keys))
            elif model_name == 'llama':
                result.update(_call_llama(prompt, api_keys))
            else:
                raise ValueError(f"Unknown model: {model_name}")

            # Extract estimated value
            if result['response']:
                result['estimated_value'] = extract_price_from_text(result['response'])

            break  # Success, exit retry loop

        except Exception as e:
            result['error'] = str(e)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"❌ Failed after {max_retries} attempts: {e}")

    result['latency'] = time.time() - start_time
    return result


def _call_claude(prompt: str, api_keys: Dict[str, str]) -> Dict:
    """Call Claude API"""
    import anthropic

    client = anthropic.Anthropic(api_key=api_keys.get('claude'))

    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=2000,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}]
    )

    response_text = message.content[0].text
    tokens = message.usage.input_tokens + message.usage.output_tokens

    return {
        'response': response_text,
        'tokens': tokens,
        'raw_response': message
    }


def _call_mistral(prompt: str, api_keys: Dict[str, str]) -> Dict:
    """Call Mistral API"""
    from mistralai import Mistral

    with Mistral(api_key=api_keys.get('mistral')) as client:
        response = client.chat.complete(
            model="mistral-large-2411",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2000
        )

    response_text = response.choices[0].message.content
    tokens = response.usage.total_tokens if hasattr(response, 'usage') else None

    return {
        'response': response_text,
        'tokens': tokens,
        'raw_response': response
    }


def _call_llama(prompt: str, api_keys: Dict[str, str]) -> Dict:
    """Call Llama via OpenRouter"""
    import requests

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_keys.get('llama')}",
            "Content-Type": "application/json",
        },
        json={
            "model": "meta-llama/llama-3.3-70b-instruct:free",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 2000
        }
    )

    if response.status_code != 200:
        raise Exception(f"OpenRouter API error: {response.status_code} - {response.text}")

    data = response.json()
    response_text = data['choices'][0]['message']['content']
    tokens = data.get('usage', {}).get('total_tokens')

    return {
        'response': response_text,
        'tokens': tokens,
        'raw_response': data
    }


# ============================================================================
# TEST EXECUTION
# ============================================================================

def run_single_home_test(
    home_data: Dict,
    actual_price: int,
    comparables: List[Dict],
    api_keys: Dict[str, str]
) -> pd.DataFrame:
    """
    Run all 15 experimental conditions (5 prompts × 3 models) on a single home

    Args:
        home_data: Dictionary with home features
        actual_price: Actual sale price
        comparables: List of 3 comparable homes for few-shot prompting
        api_keys: API keys for all models

    Returns:
        DataFrame with all results
    """
    models = ['claude', 'mistral', 'llama']
    strategies = list(PROMPT_STRATEGIES.keys())

    results = []
    total_calls = len(models) * len(strategies)
    current_call = 0

    print(f"\n{'='*80}")
    print(f"Testing Home: {home_data['bedrooms']}BR/{home_data['bathrooms']}BA, {home_data['lot_size']}sqft, Built {home_data['year_built']}")
    print(f"Actual Sale Price: ${actual_price:,}")
    print(f"Total API Calls: {total_calls}")
    print(f"{'='*80}\n")

    for strategy in strategies:
        # Format prompt
        if strategy == 'few_shot':
            prompt = format_prompt(strategy, home_data, comparables)
        else:
            prompt = format_prompt(strategy, home_data)

        print(f"\n📝 Strategy: {strategy.upper().replace('_', ' ')}")
        print("-" * 80)

        for model in models:
            current_call += 1
            print(f"  [{current_call}/{total_calls}] Calling {model}...", end=" ")

            result = call_llm(
                model_name=model,
                prompt=prompt,
                home_id=home_data.get('zpid', 'test_home'),
                strategy_name=strategy,
                api_keys=api_keys
            )

            result['actual_price'] = actual_price

            if result['estimated_value']:
                error = result['estimated_value'] - actual_price
                pct_error = (error / actual_price) * 100
                result['absolute_error'] = abs(error)
                result['percent_error'] = pct_error
                print(f"✅ ${result['estimated_value']:,} ({pct_error:+.1f}%)")
            else:
                result['absolute_error'] = None
                result['percent_error'] = None
                print(f"⚠️ Failed to extract value")

            results.append(result)

            # Rate limiting
            time.sleep(1)

    return pd.DataFrame(results)


def analyze_results(df: pd.DataFrame) -> Dict:
    """
    Analyze test results and generate summary statistics

    Args:
        df: DataFrame with test results

    Returns:
        Dictionary with analysis
    """
    analysis = {}

    # Filter successful extractions
    df_valid = df[df['estimated_value'].notna()].copy()

    if len(df_valid) == 0:
        return {"error": "No valid price extractions"}

    # Overall statistics
    analysis['overall'] = {
        'total_calls': len(df),
        'successful_extractions': len(df_valid),
        'extraction_rate': len(df_valid) / len(df) * 100,
        'mean_estimate': df_valid['estimated_value'].mean(),
        'median_estimate': df_valid['estimated_value'].median(),
        'std_estimate': df_valid['estimated_value'].std(),
        'min_estimate': df_valid['estimated_value'].min(),
        'max_estimate': df_valid['estimated_value'].max(),
        'mean_absolute_error': df_valid['absolute_error'].mean(),
        'mean_percent_error': df_valid['percent_error'].mean()
    }

    # By model
    analysis['by_model'] = df_valid.groupby('model').agg({
        'estimated_value': ['mean', 'std', 'count'],
        'absolute_error': 'mean',
        'percent_error': 'mean'
    }).to_dict()

    # By strategy
    analysis['by_strategy'] = df_valid.groupby('strategy').agg({
        'estimated_value': ['mean', 'std', 'count'],
        'absolute_error': 'mean',
        'percent_error': 'mean'
    }).to_dict()

    return analysis


def create_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a comparison table for visualization

    Args:
        df: DataFrame with results

    Returns:
        Pivot table with strategies vs models
    """
    # Filter valid estimates
    df_valid = df[df['estimated_value'].notna()].copy()

    # Create pivot table
    pivot = df_valid.pivot_table(
        values='estimated_value',
        index='strategy',
        columns='model',
        aggfunc='first'
    )

    # Add actual price column
    if len(df_valid) > 0:
        pivot['actual'] = df_valid['actual_price'].iloc[0]

    return pivot


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("LLM HOME VALUATION - SINGLE HOME TEST")
    print("="*80)

    # Test home data (from notebook: home at index 2)
    test_home = {
        'zpid': '11469598',
        'bedrooms': 3,
        'bathrooms': 2,
        'lot_size': 6695,
        'year_built': 1958
    }

    actual_price = 207000

    # Comparables for few-shot (from notebook data)
    comparables = [
        {'bathrooms': 2, 'lot_size': 6000, 'year_built': 1957, 'price': 205000},
        {'bathrooms': 2, 'lot_size': 8450, 'year_built': 1958, 'price': 215000},
        {'bathrooms': 2, 'lot_size': 8925, 'year_built': 1960, 'price': 230000}
    ]

    # Load API keys from environment
    api_keys = {
        'claude': os.getenv('ANTHROPIC_API_KEY'),
        'mistral': os.getenv('MISTRAL_API_KEY'),
        'llama': os.getenv('OPENROUTER_API_KEY', os.getenv('LLAMA_API_KEY'))
    }

    # Validate API keys
    missing_keys = [k for k, v in api_keys.items() if not v]
    if missing_keys:
        print(f"\n⚠️ WARNING: Missing API keys for: {', '.join(missing_keys)}")
        print("Set environment variables: ANTHROPIC_API_KEY, MISTRAL_API_KEY, OPENROUTER_API_KEY")
        print("\nProceeding with available models only...\n")

    # Run test
    results_df = run_single_home_test(test_home, actual_price, comparables, api_keys)

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f'test_results_{timestamp}.csv'
    results_df.to_csv(results_file, index=False)
    print(f"\n💾 Results saved to: {results_file}")

    # Create comparison table
    comparison = create_comparison_table(results_df)
    print("\n" + "="*80)
    print("COMPARISON TABLE: Estimated Values by Strategy and Model")
    print("="*80)
    print(comparison.to_string())

    # Analyze results
    analysis = analyze_results(results_df)

    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)

    if 'error' in analysis:
        print(f"\n⚠️ {analysis['error']}")
    else:
        print(f"\nTotal API Calls: {analysis['overall']['total_calls']}")
        print(f"Successful Extractions: {analysis['overall']['successful_extractions']} ({analysis['overall']['extraction_rate']:.1f}%)")
        print(f"\nEstimate Range: ${analysis['overall']['min_estimate']:,} - ${analysis['overall']['max_estimate']:,}")
        print(f"Mean Estimate: ${analysis['overall']['mean_estimate']:,.0f}")
        print(f"Median Estimate: ${analysis['overall']['median_estimate']:,.0f}")
        print(f"Std Deviation: ${analysis['overall']['std_estimate']:,.0f}")
        print(f"\nMean Absolute Error: ${analysis['overall']['mean_absolute_error']:,.0f}")
        print(f"Mean Percent Error: {analysis['overall']['mean_percent_error']:.1f}%")
        print(f"Actual Price: ${actual_price:,}")

    # Save analysis
    analysis_file = f'test_analysis_{timestamp}.json'
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"\n💾 Analysis saved to: {analysis_file}")

    print("\n" + "="*80)
    print("TEST COMPLETE!")
    print("="*80)
