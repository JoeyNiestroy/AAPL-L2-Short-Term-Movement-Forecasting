# AAPL-L2-Short-Term-Movement-Forecasting

This project investigates whether tick level limit order book data can reliable predict the direction of Apple (`AAPL`) at various short time horizons

The pipeline processes Databento XNAS MBP-10 data without resampling. Code is mainly claude / chat slop, will be refactored if perfomance is promising

## Pipeline

1. Download and convert MBP-10 data to daily Parquet files ( 	dbn_to_parquet.py ) .
2. Restrict observations to regular trading hours.
3. Extract 170 tick-level features ( extract_tick_features.py , provided bc it didn't work lol ), including:

   * Bid-ask spread and microprice
   * Depth and order-count imbalance
   * Price gaps across ten book levels
   * Event type, side, size, and depth
   * Tick-to-tick book changes
   * Rolling order-flow and volatility measures
4. Generate three-class labels:

   * `-1`: meaningful downward movement
   * `0`: no meaningful movement
   * `1`: meaningful upward movement
5. Evaluate several classifiers

Labels incorporate an assumed latency of 1,000 market events or around 50 ms. The test models allow for sub 25ms inference time. Open bid / asks are assumed vs midprice

## Dataset

* **Instrument:** AAPL
* **Market:** Nasdaq
* **Data:** Databento MBP-10
* **Daily files:** 502
* **Date range:** November 2023–November 2025
* **Valid training ticks:** 527 million
* **Sampled training rows:** 7.02 million
* **Validation rows:** 1.50 million
* **Test rows:** 1.52 million
* **Features:** 170

Files are split chronologically to prevent information leaks.

## Preliminary Results — 60-Minute Horizon

| Model                       | Balanced Accuracy |   Macro F1 | Mean Signal Return |
| --------------------------- | ----------------: | ---------: | -----------------: |
| Logistic Regression         |        **0.3804** |     0.3307 |          -1.17 bps |
| XGBoost                     |            0.3562 | **0.3514** |          -1.49 bps |
| LightGBM                    |            0.3519 |     0.3448 |          -2.53 bps |
| Random Forest               |            0.3468 |     0.3383 |          -1.29 bps |
| Histogram Gradient Boosting |            0.3442 |     0.3417 |          -2.51 bps |
| Majority-Class Baseline     |            0.3333 |     0.2045 |          -0.05 bps |

The models detect some out-of-sample structure relative to the balanced-accuracy baseline, but none of the initial strategies produce positive average executable returns. 
## Preliminary Results — 20-Minute Horizon

| Model                       | Balanced Accuracy |   Macro F1 |   Log Loss | Signal Rate | Mean Signal Return | Signal Win Rate |
| --------------------------- | ----------------: | ---------: | ---------: | ----------: | -----------------: | --------------: |
| Logistic Regression         |        **0.3988** |     0.3588 | **1.0694** |      59.46% |          -1.58 bps |      **48.73%** |
| LightGBM                    |            0.3888 |     0.3463 |     1.1298 |      58.85% |          -2.12 bps |          47.37% |
| XGBoost                     |            0.3884 |     0.3588 |     1.1382 |      64.30% |          -1.73 bps |          47.74% |
| Random Forest               |            0.3809 | **0.3601** |     1.1684 |      71.89% |          -1.76 bps |          47.90% |
| Histogram Gradient Boosting |            0.3705 |     0.3529 |     1.2191 |      68.31% |          -2.21 bps |          46.59% |
| Majority-Class Baseline     |            0.3333 |     0.1935 |     1.0507 |     100.00% |          -0.03 bps |          48.26% |

The 20-minute models show stronger classification performance than the initial one-hour models. Logistic regression achieves the highest balanced accuracy, while random forest produces the highest macro F1. However, all model-generated signals still have negative average executable returns and win rates below 50%, so still no viable trading opportunities. 

## Next Steps

I plan to test across a variety of horizons, as well as expanding and refining feature set.  
