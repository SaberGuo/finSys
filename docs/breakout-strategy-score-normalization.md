# Breakout Strategy Score Normalization Fix

## Issue

**Error**: `All scores must be in range [-1.0, 1.0]`

**Date Reported**: 2024-09-27

**Root Cause**: The scoring function in `BreakoutStrategy._score_candidates()` was calculating raw scores that could exceed 1.0:

```python
# Old scoring logic
breakout_score = (close - ma) / ma  # Can be > 1.0 for large breakouts
vol_strength = (volume - vol_ma) / vol_ma  # Can be > 1.0 for volume surges
score = 0.6 * breakout_score + 0.4 * vol_strength  # Combined can exceed 1.0
```

**Example**: 
- Stock price 50% above MA: `breakout_score = 0.50`
- Volume 5x average: `vol_strength = 4.0`
- Combined: `0.6 * 0.50 + 0.4 * 4.0 = 1.90` ❌ (exceeds 1.0)

## Solution

Applied **tanh normalization** to map raw scores to [-1.0, 1.0] range:

```python
# New scoring logic
raw_score = 0.6 * breakout_score + 0.4 * vol_strength
normalized_score = tanh(raw_score * 5)  # Maps to [-1.0, 1.0]
```

### Why tanh?

1. **Bounded output**: `tanh(x)` always returns values in (-1, 1)
2. **Preserves ranking**: Monotonic function maintains relative order
3. **Smooth mapping**: Typical values (0.05-0.5) map to reasonable scores (0.24-0.99)
4. **Handles extremes**: Very large values asymptotically approach 1.0

### Score Mapping Examples

| Raw Score | Description | Normalized Score |
|-----------|-------------|------------------|
| 0.05 | Small breakout (5%) | 0.2449 |
| 0.10 | Medium breakout (10%) | 0.4621 |
| 0.50 | Large breakout (50%) | 0.9866 |
| 1.00 | Very large (100%) | 0.9999 |
| 1.90 | Extreme case | 1.0000 |

## Changes Made

### Code Changes

**File**: `finquant/selection/strategies/breakout.py`

**Method**: `_score_candidates()` (lines 306-340)

```python
# Added normalization step
normalized_score = float(np.tanh(raw_score * 5))
scores[tic] = normalized_score
```

### Documentation Updates

**File**: `docs/breakout-strategy-guide.md`

Added section explaining score normalization:
- Raw score calculation
- Normalization formula
- Typical value mappings

## Verification

### Unit Tests
- ✅ All 19 unit tests passing
- ✅ All 13 integration tests passing

### Synthetic Test
```python
# Extreme case: 50% breakout + 5x volume
raw_score = 1.90
normalized = tanh(1.90 * 5) = 1.000000  ✓
```

### Real Data Test
```bash
finquant selection run \
  --config config/selection_breakout_test.yaml \
  --start 2020-08-01 --end 2020-12-31
```
Result: All scores within [-1.0, 1.0] ✓

## Impact

- **Backward compatible**: Relative rankings preserved
- **No config changes needed**: Works with existing configurations
- **Performance**: Negligible overhead (vectorized numpy operation)

## Related Files

- `finquant/selection/strategies/breakout.py` - Implementation
- `docs/breakout-strategy-guide.md` - User documentation
- `tests/unit/test_breakout_strategy.py` - Unit tests
- `tests/integration/test_breakout_selection.py` - Integration tests

## Date Fixed

2026-05-02
