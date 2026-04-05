# Probability Forecasting - Step-by-Step Example

This document explains how probability forecasts are created using a simple, real-world example. We'll walk through each step with actual numbers so you can see exactly how it works.

---

## The Setup

**Store**: Store1  
**Item**: ItemA  
**Historical Period**: 400 days (Jan 1, 2023 - Feb 4, 2024)

**What we have:**
- Historical sales data (actual sales for the past 400 days)
- Point forecasts (predictions for the next 30 days)

**What we want:**
- Quantile forecasts (probability ranges) for the next 30 days

**Sample Historical Sales** (first 10 days):
| Day | Date | Actual Sales |
|-----|------|--------------|
| 1 | 2023-01-01 | 50 |
| 2 | 2023-01-02 | 45 |
| 3 | 2023-01-03 | 55 |
| 4 | 2023-01-04 | 60 |
| 5 | 2023-01-05 | 48 |
| 6 | 2023-01-06 | 52 |
| 7 | 2023-01-07 | 58 |
| 8 | 2023-01-08 | 62 |
| 9 | 2023-01-09 | 49 |
| 10 | 2023-01-10 | 53 |
| ... | ... | ... |
| 400 | 2024-02-04 | 54 |

**Future Forecast Period**: 30 days (Feb 5, 2024 - Mar 6, 2024)

**Point Forecasts** (first 5 days):
| Day | Date | Point Forecast |
|-----|------|-----------------|
| 401 | 2024-02-05 | 55.0 |
| 402 | 2024-02-06 | 58.0 |
| 403 | 2024-02-07 | 60.0 |
| 404 | 2024-02-08 | 57.0 |
| 405 | 2024-02-09 | 59.0 |

---

## Step 1: Extract the Trend

**What is a trend?** The long-term pattern in sales (is it going up, down, or staying flat?).

**How we calculate it:** We use a rolling average (average of nearby days) to smooth out daily fluctuations.

**Window Size Calculation:**
- We use a window of 200 days (half of our 400-day history)
- Formula: `w = min(365, floor(400/2)) = min(365, 200) = 200`

**What this means:** For each day, we look at 200 days around it and take the average. This smooths out the noise and shows the underlying trend.

**Example Trend Values:**
- Day 1: Trend = 52.0
- Day 2: Trend = 51.5
- Day 3: Trend = 53.0
- Day 100: Trend = 52.5
- Day 400: Trend = 55.2

---

## Step 2: Extract Seasonal Patterns

**What is seasonality?** Patterns that repeat every year (e.g., higher sales in December, lower sales in January).

**How we calculate it:** Since we have more than 365 days of data, we can find patterns for each day of the year.

**Day 0 (January 1) Calculation:**

We look at all January 1st dates in our history:
- 2023-01-01 (Day 1): Actual = 50, Trend = 52.0
  - Difference = 50 - 52.0 = -2.0
- 2024-01-01 (Day 366): Actual = 55, Trend = 54.0
  - Difference = 55 - 54.0 = +1.0

**Seasonal Component for January 1:**
- Average of differences: (-2.0 + 1.0) / 2 = -0.5

**What this means:** On average, January 1st sales are 0.5 units below the trend.

**Day 100 (April 10) Calculation:**

Only one observation:
- 2023-04-10 (Day 100): Actual = 60, Trend = 52.5
  - Difference = 60 - 52.5 = +7.5

**Seasonal Component for April 10:**
- S₁₀₀ = +7.5

**What this means:** April 10th typically has sales 7.5 units above the trend.

**Day 364 (December 31) Calculation:**

Only one observation:
- 2023-12-31 (Day 365): Actual = 70, Trend = 68.0
  - Difference = 70 - 68.0 = +2.0

**Seasonal Component for December 31:**
- S₃₆₄ = +2.0

---

## Step 3: Calculate Residuals

**What is a residual?** The leftover part after we remove trend and seasonality. It represents the unpredictable "noise" in the data.

**Formula:** `Residual = Actual Sales - Trend - Seasonality`

**Example Calculations:**

**Day 1 (2023-01-01, Day-of-Year = 0):**
- Actual = 50
- Trend = 52.0
- Seasonality (Day 0) = -0.5
- **Residual = 50 - 52.0 - (-0.5) = 50 - 52.0 + 0.5 = -1.5**

**Day 100 (2023-04-10, Day-of-Year = 99):**
- Actual = 60
- Trend = 52.5
- Seasonality (Day 99) = 7.5
- **Residual = 60 - 52.5 - 7.5 = 0.0**

**Day 365 (2023-12-31, Day-of-Year = 364):**
- Actual = 70
- Trend = 68.0
- Seasonality (Day 364) = 2.0
- **Residual = 70 - 68.0 - 2.0 = 0.0**

**What residuals tell us:** 
- If residual is positive: actual sales were higher than expected (after accounting for trend and seasonality)
- If residual is negative: actual sales were lower than expected
- If residual is zero: actual sales matched expectations perfectly

---

## Step 4: Calculate Statistics from Residuals

Now we analyze all the residuals to understand:
1. **Bias**: Does the model consistently over-predict or under-predict?
2. **Uncertainty**: How much variation is there in the residuals?

### 4.1 Mean Residual (Bias Correction)

**What it measures:** The average error. If it's negative, the model tends to over-predict. If positive, it under-predicts.

**Calculation:**
- We have 380 valid residuals (some are NaN at the edges)
- Mean = Sum of all residuals / Number of residuals

**Result:** μᵣ = -0.2

**What this means:** On average, the model over-predicts by 0.2 units. So we need to subtract 0.2 from our forecasts to correct for this bias.

### 4.2 Overall Standard Deviation

**What it measures:** How spread out the residuals are. A larger standard deviation means more uncertainty.

**Result:** σᵣ = 3.5

**What this means:** The typical forecast error is about 3.5 units. Most residuals fall within ±3.5 units of the mean.

### 4.3 Day-of-Year Standard Deviations

**What it measures:** Some days of the year might have more uncertainty than others. We calculate a separate standard deviation for each day of the year.

**Day 0 (January 1):**

Residuals for January 1:
- r₁ = -1.5 (from 2023-01-01)
- r₃₆₆ = +1.0 (from 2024-01-01)

**Mean for Day 0:** (-1.5 + 1.0) / 2 = -0.25

**Standard Deviation for Day 0:**
- σᵣ,₀ = √[((-1.5 - (-0.25))² + (1.0 - (-0.25))²) / 1]
- σᵣ,₀ = √[(-1.25)² + (1.25)²] = √[1.5625 + 1.5625] = √3.125 = **1.768**

**Day 100 (April 10):**

Only one observation, so we use the overall standard deviation:
- σᵣ,₁₀₀ = 3.5

**Day 364 (December 31):**

Only one observation, so we use the overall standard deviation:
- σᵣ,₃₆₄ = 3.5

**Complete Array:** We have 365 values (one for each day of the year), for example:
- σᵣ,₀ = 1.768
- σᵣ,₃₅ = 3.2 (for February 5)
- σᵣ,₃₆ = 3.1 (for February 6)
- ... and so on

---

## Step 5: Generate Quantile Forecasts

Now we use all this information to create probability forecasts for a future date.

**Target Date**: 2024-02-05 (Day 401, Day-of-Year = 35)  
**Point Forecast**: 55.0

### 5.1 Get Day-Specific Standard Deviation

For February 5 (Day-of-Year = 35):
- σᵣ,₃₅ = 3.2

### 5.2 Understand Z-Scores

**What is a z-score?** It tells us how many standard deviations away from the mean we need to go to capture a certain probability.

**Common Z-Scores:**
- z₀.₀₅ = -1.645 (for 5th percentile - very low)
- z₀.₁₀ = -1.282 (for 10th percentile - low)
- z₀.₂₅ = -0.675 (for 25th percentile - below median)
- z₀.₅₀ = 0.000 (for 50th percentile - median)
- z₀.₇₅ = +0.675 (for 75th percentile - above median)
- z₀.₉₀ = +1.282 (for 90th percentile - high)
- z₀.₉₅ = +1.645 (for 95th percentile - very high)

**What this means:** 
- Negative z-scores give us lower forecasts (pessimistic scenarios)
- Positive z-scores give us higher forecasts (optimistic scenarios)
- Zero gives us the median (most likely scenario)

### 5.3 Calculate Quantile Forecasts

**The Formula:**
```
Quantile Forecast = Point Forecast + Bias Correction + (Z-Score × Day-Specific Std Dev)
```

**In our case:**
- Point Forecast = 55.0
- Bias Correction (μᵣ) = -0.2
- Day-Specific Std Dev (σᵣ,₃₅) = 3.2

**5th Percentile (Q₀.₀₅) - Very Pessimistic:**
```
Q₀.₀₅ = 55.0 + (-0.2) + (-1.645) × 3.2
Q₀.₀₅ = 55.0 - 0.2 - 5.264
Q₀.₀₅ = 49.536 ≈ 49.5
```
**Meaning:** There's a 5% chance actual sales will be 49.5 or lower.

**10th Percentile (Q₀.₁₀) - Pessimistic:**
```
Q₀.₁₀ = 55.0 + (-0.2) + (-1.282) × 3.2
Q₀.₁₀ = 55.0 - 0.2 - 4.102
Q₀.₁₀ = 50.698 ≈ 50.7
```
**Meaning:** There's a 10% chance actual sales will be 50.7 or lower.

**25th Percentile (Q₀.₂₅) - Below Median:**
```
Q₀.₂₅ = 55.0 + (-0.2) + (-0.675) × 3.2
Q₀.₂₅ = 55.0 - 0.2 - 2.160
Q₀.₂₅ = 52.640 ≈ 52.6
```
**Meaning:** There's a 25% chance actual sales will be 52.6 or lower.

**50th Percentile / Median (Q₀.₅₀) - Most Likely:**
```
Q₀.₅₀ = 55.0 + (-0.2) + (0.000) × 3.2
Q₀.₅₀ = 55.0 - 0.2 + 0.0
Q₀.₅₀ = 54.800 = 54.8
```
**Meaning:** There's a 50% chance actual sales will be 54.8 or lower (and 50% chance it will be higher). This is the median forecast.

**75th Percentile (Q₀.₇₅) - Above Median:**
```
Q₀.₇₅ = 55.0 + (-0.2) + (+0.675) × 3.2
Q₀.₇₅ = 55.0 - 0.2 + 2.160
Q₀.₇₅ = 56.960 ≈ 57.0
```
**Meaning:** There's a 75% chance actual sales will be 57.0 or lower.

**90th Percentile (Q₀.₉₀) - Optimistic:**
```
Q₀.₉₀ = 55.0 + (-0.2) + (+1.282) × 3.2
Q₀.₉₀ = 55.0 - 0.2 + 4.102
Q₀.₉₀ = 58.902 ≈ 58.9
```
**Meaning:** There's a 90% chance actual sales will be 58.9 or lower.

**95th Percentile (Q₀.₉₅) - Very Optimistic:**
```
Q₀.₉₅ = 55.0 + (-0.2) + (+1.645) × 3.2
Q₀.₉₅ = 55.0 - 0.2 + 5.264
Q₀.₉₅ = 60.064 ≈ 60.1
```
**Meaning:** There's a 95% chance actual sales will be 60.1 or lower.

---

## Complete Forecast Table

**For Date: 2024-02-05 (Day-of-Year = 35)**

| Quantile | Z-Score | Calculation | Forecast | Interpretation |
|----------|---------|------------|----------|-----------------|
| q05 | -1.645 | 55.0 - 0.2 - 1.645 × 3.2 | **49.5** | 5% chance sales ≤ 49.5 |
| q10 | -1.282 | 55.0 - 0.2 - 1.282 × 3.2 | **50.7** | 10% chance sales ≤ 50.7 |
| q25 | -0.675 | 55.0 - 0.2 - 0.675 × 3.2 | **52.6** | 25% chance sales ≤ 52.6 |
| q50 | 0.000 | 55.0 - 0.2 + 0.000 × 3.2 | **54.8** | 50% chance sales ≤ 54.8 (median) |
| q75 | +0.675 | 55.0 - 0.2 + 0.675 × 3.2 | **57.0** | 75% chance sales ≤ 57.0 |
| q90 | +1.282 | 55.0 - 0.2 + 1.282 × 3.2 | **58.9** | 90% chance sales ≤ 58.9 |
| q95 | +1.645 | 55.0 - 0.2 + 1.645 × 3.2 | **60.1** | 95% chance sales ≤ 60.1 |

**For Date: 2024-02-06 (Day-of-Year = 36)**

Point Forecast: 58.0  
Day-Specific Std Dev: 3.1

| Quantile | Calculation | Forecast | Interpretation |
|----------|------------|----------|-----------------|
| q05 | 58.0 - 0.2 - 1.645 × 3.1 | **52.1** | 5% chance sales ≤ 52.1 |
| q10 | 58.0 - 0.2 - 1.282 × 3.1 | **53.2** | 10% chance sales ≤ 53.2 |
| q25 | 58.0 - 0.2 - 0.675 × 3.1 | **55.1** | 25% chance sales ≤ 55.1 |
| q50 | 58.0 - 0.2 + 0.000 × 3.1 | **57.8** | 50% chance sales ≤ 57.8 (median) |
| q75 | 58.0 - 0.2 + 0.675 × 3.1 | **59.9** | 75% chance sales ≤ 59.9 |
| q90 | 58.0 - 0.2 + 1.282 × 3.1 | **61.8** | 90% chance sales ≤ 61.8 |
| q95 | 58.0 - 0.2 + 1.645 × 3.1 | **62.9** | 95% chance sales ≤ 62.9 |

---

## Key Takeaways

1. **Point Forecast (55.0)**: This is the single "best guess" prediction.

2. **Quantile Forecasts**: These give us a range of possible outcomes with probabilities:
   - **q05 to q95**: This range (49.5 to 60.1) covers 90% of possible outcomes
   - **q10 to q90**: This range (50.7 to 58.9) covers 80% of possible outcomes
   - **q50 (54.8)**: This is the median - equally likely to be above or below

3. **Why the median (q50) is different from point forecast (55.0)?**
   - The point forecast is 55.0
   - But we found a bias of -0.2 (model over-predicts)
   - So the corrected median is 55.0 - 0.2 = 54.8

4. **Confidence Intervals:**
   - **80% Confidence Interval**: [q10, q90] = [50.7, 58.9] - 80% of actual sales will fall in this range
   - **90% Confidence Interval**: [q05, q95] = [49.5, 60.1] - 90% of actual sales will fall in this range

5. **The Process:**
   - Extract trend and seasonality from historical data
   - Calculate residuals (unpredictable part)
   - Analyze residuals to find bias and uncertainty
   - Use this information to adjust point forecasts into probability ranges

---

## Summary

This example showed how we:
1. Started with historical sales data and point forecasts
2. Extracted trend and seasonal patterns
3. Calculated residuals to understand past errors
4. Used residual statistics to measure bias and uncertainty
5. Generated quantile forecasts that show probability ranges

The result is a complete picture of uncertainty around each forecast, not just a single number!
