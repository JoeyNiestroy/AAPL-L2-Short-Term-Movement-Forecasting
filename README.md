# AAPL-L2-Short-Term-Movement-Forecasting

This project investigates whether tick level limit order book data can reliable predict the direction of Apple (`AAPL`) approximately one hour into the future.

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

## Preliminary Results

| Model                       | Balanced Accuracy |   Macro F1 | Mean Signal Return |
| --------------------------- | ----------------: | ---------: | -----------------: |
| Logistic Regression         |        **0.3804** |     0.3307 |          -1.17 bps |
| XGBoost                     |            0.3562 | **0.3514** |          -1.49 bps |
| LightGBM                    |            0.3519 |     0.3448 |          -2.53 bps |
| Random Forest               |            0.3468 |     0.3383 |          -1.29 bps |
| Histogram Gradient Boosting |            0.3442 |     0.3417 |          -2.51 bps |
| Majority-Class Baseline     |            0.3333 |     0.2045 |          -0.05 bps |

The models detect some out-of-sample structure relative to the balanced-accuracy baseline, but none of the initial strategies produce positive average executable returns. 

## Next Steps

I plan to test across a variety of horizons, as well as expanding and refining feature set.  
