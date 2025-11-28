"""
Quick Test Runner for LLM Valuation Pipeline
Author: Anshul Kumar (anshulk@andrew.cmu.edu)

This script provides a simplified interface to run the valuation test.
"""

import os
import sys

def setup_environment():
    """
    Check and setup environment variables for API keys
    """
    print("Checking API Key Configuration...")
    print("-" * 60)

    required_keys = {
        'ANTHROPIC_API_KEY': 'Claude (Anthropic)',
        'MISTRAL_API_KEY': 'Mistral',
        'OPENROUTER_API_KEY': 'Llama (via OpenRouter)'
    }

    available = []
    missing = []

    for env_var, service_name in required_keys.items():
        if os.getenv(env_var):
            print(f"✅ {service_name}: API key found")
            available.append(service_name)
        else:
            print(f"❌ {service_name}: API key NOT found (set {env_var})")
            missing.append(service_name)

    print("-" * 60)

    if missing:
        print(f"\n⚠️  Missing API keys for: {', '.join(missing)}")
        print("\nTo set API keys, run:")
        for env_var in required_keys.keys():
            if env_var not in [k for k in required_keys.keys() if os.getenv(k)]:
                print(f"  export {env_var}='your_api_key_here'")
        print("\nProceeding with available models only...")

    return len(available) > 0


def run_test():
    """
    Run the single home test
    """
    if not setup_environment():
        print("\n❌ No API keys configured. Cannot proceed.")
        sys.exit(1)

    print("\n" + "="*80)
    print("Starting LLM Valuation Test...")
    print("="*80)

    # Import and run the main pipeline
    from llm_valuation_pipeline import run_single_home_test, create_comparison_table, analyze_results
    import pandas as pd
    from datetime import datetime

    # Test home data
    test_home = {
        'zpid': '11469598',
        'bedrooms': 3,
        'bathrooms': 2,
        'lot_size': 6695,
        'year_built': 1958
    }

    actual_price = 207000

    # Comparables
    comparables = [
        {'bathrooms': 2, 'lot_size': 6000, 'year_built': 1957, 'price': 205000},
        {'bathrooms': 2, 'lot_size': 8450, 'year_built': 1958, 'price': 215000},
        {'bathrooms': 2, 'lot_size': 8925, 'year_built': 1960, 'price': 230000}
    ]

    # API keys
    api_keys = {
        'claude': os.getenv('ANTHROPIC_API_KEY'),
        'mistral': os.getenv('MISTRAL_API_KEY'),
        'llama': os.getenv('OPENROUTER_API_KEY')
    }

    # Run test
    try:
        results_df = run_single_home_test(test_home, actual_price, comparables, api_keys)

        # Display results
        comparison = create_comparison_table(results_df)
        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        print(comparison.to_string())

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f'test_results_{timestamp}.csv'
        results_df.to_csv(results_file, index=False)
        print(f"\n✅ Results saved to: {results_file}")

        print("\n" + "="*80)
        print("TEST COMPLETE!")
        print("="*80)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_test()
