# Steering Vector + Token Activation Research Report

**Experiment ID:** 20250801_195604
**Generated:** 2025-08-01 19:56:09

## Configuration

- **Model:** meta-llama/Llama-3.1-8B-Instruct
- **Top Layers:** 5
- **Tokens per Layer:** 10
- **Prompt Strategies:** natural, template, context

## Results Summary

### Diagnostics
**Status:** completed

### Combination
**Status:** completed

### Prompt_Generation
**Status:** completed

## Key Findings

1. Identified 5 strongest harmful steering layers
2. Layer -13 shows the highest steering magnitude

## Recommendations

1. Test with a larger dataset to improve token selection reliability
2. Experiment with different prompt templates and strategies
3. Consider using multiple tokens simultaneously for stronger effects
4. Validate results with human evaluation of generated responses

## Next Steps

1. Run full evaluation with model inference to measure actual steering effects
2. Test the approach on different types of harmful content beyond the current dataset
3. Compare results with traditional activation patching methods
4. Investigate why certain layers show stronger steering effects
5. Develop automated metrics for measuring steering effectiveness
6. Scale up the experiment with more layers and token combinations
