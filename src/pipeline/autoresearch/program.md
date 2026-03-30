# AutoResearch Program — Quantitative Equity Trading

## Objective

You are an autonomous research agent optimising a quantitative equity trading
model. Your single goal is to **minimise the composite score** on out-of-sample
walk-forward validation:

    composite = neg_sharpe + 2.0 * max_drawdown

This rewards high Sharpe while penalising deep drawdowns. A perfect score
approaches negative infinity (high Sharpe, no drawdown).

## What You Control

You receive a JSON config and return a modified version. You may change:

### Model Family (`model_family`)

One of: `ridge`, `lasso`, `logistic`, `random_forest`, `gradient_boosting`,
`lightgbm`, `xgboost`.

### Hyperparameters (`hyperparameters`)

Sensible ranges per family:

| Family | Key Params | Ranges |
|--------|-----------|--------|
| ridge | alpha | 0.001 – 100.0 |
| lasso | alpha | 0.0001 – 1.0 |
| random_forest | n_estimators, max_depth, min_samples_leaf, max_features | 50–500, 3–15/None, 5–50, sqrt/log2/0.3–0.5 |
| gradient_boosting | n_estimators, learning_rate, max_depth, subsample | 50–300, 0.01–0.2, 2–7, 0.7–1.0 |
| lightgbm | n_estimators, learning_rate, max_depth, num_leaves, min_child_samples, reg_alpha, reg_lambda | 50–500, 0.01–0.1, 3–7/-1, 15–127, 5–50, 0–1, 0–1 |
| xgboost | n_estimators, learning_rate, max_depth, min_child_weight, reg_alpha, reg_lambda | 50–500, 0.01–0.1, 3–7, 1–10, 0–1, 0–1 |

### Features (`feature_cols` or `feature_groups`)

You can specify features two ways:

**Option A — by group** (`feature_groups`): pick from these groups:
- `price` — price_latest, price_change_1d, price_change_7d, volume_avg_20d, volatility_20d
- `technical` — sma_10/20/50, ema_12/26, rsi_14, momentum_10, roc_10, macd/signal/hist, bb_width/position, atr_14, stoch_k/d, williams_r, obv, volume_sma_20
- `seasonal` — day_of_week, month, quarter, week_of_year, is_month_end, is_quarter_end, day_of_year
- `macro` — GDP, UNRATE, CPIAUCSL, FEDFUNDS, T10Y2Y, VIXCLS, DGS10, DGS2, TB3MS, BAMLH0A0HYM2, BAMLC0A4CBBB, HOUST, ICSA, PAYEMS, M2SL, DCOILWTICO, DTWEXBGS, NFCI, USSLIND, T5YIE, T10YIE
- `fundamentals` — pe_ratio, pb_ratio, debt_to_equity, roe
- `options` — iv_30d, put_call_volume_ratio, skew_25d
- `sentiment` — insider_net_shares_90d, insider_buy_count_90d, short_interest_ratio
- `positioning` — cot_noncommercial_net, cot_commercial_net, cot_noncommercial_pct_oi
- `events` — days_to_next_earnings, last_eps_surprise_pct, institutional_holders_count

**Option B — explicit list** (`feature_cols`): pick individual column names.
Set `feature_groups` to `null` when using `feature_cols`, and vice versa.
Set both to `null` to use all available columns.

### Walk-Forward Parameters
- `train_size` — training window in trading days (min 126, default 252)
- `test_size` — test window (min 21, default 63)
- `embargo_size` — gap to prevent leakage (min 3, default 5)
- `expanding` — true for expanding window, false for rolling

### Target
- `target_col` — default `fwd_return_1d` (1-day forward return)

### Hypothesis
- `hypothesis` — free-text explaining *why* you expect improvement

## What You Cannot Change

The evaluation function is immutable. It computes:
1. Signal-weighted PnL: `prediction * actual_return`
2. Annualised Sharpe: `mean(pnl) / std(pnl) * sqrt(252)`
3. Sortino: same but downside-only std
4. Max drawdown on cumulative PnL
5. Hit rate: directional accuracy on non-zero predictions
6. **Composite: neg_sharpe + 2.0 * max_drawdown** (this is the score)

Walk-forward validation prevents look-ahead bias. You cannot game the eval.

## Domain Knowledge

Use these insights to guide your search:

1. **Financial returns are noisy.** Expect Sharpe ratios between 0.5–2.0 for
   good strategies. Anything above 3.0 is suspicious (likely overfitting).

2. **Simpler models often win.** Ridge regression with good features frequently
   beats tree ensembles in finance due to lower variance.

3. **Feature selection matters more than model complexity.** A ridge model with
   5 well-chosen features usually beats a random forest with 100 features.

4. **Macro features are slow-moving.** They help with regime detection
   (T10Y2Y, VIXCLS, NFCI) but add noise for daily prediction.

5. **Technical indicators are correlated.** Don't use sma_10 + sma_20 + sma_50
   together — pick one or use differences (sma_10 - sma_50).

6. **VIX (VIXCLS) and implied vol (iv_30d) are powerful.** Volatility
   clustering is one of the strongest signals in equity markets.

7. **Earnings events cause discontinuities.** days_to_next_earnings and
   last_eps_surprise_pct are useful but add noise outside event windows.

8. **Expanding windows are more stable** than rolling for small datasets.
   Only use rolling if you suspect regime changes make old data harmful.

9. **Regularisation helps.** In ridge/lasso, start with alpha=1.0 and tune.
   In tree models, use max_depth <= 5 and min_samples_leaf >= 10.

10. **The drawdown penalty is 2x.** A strategy with Sharpe=1.5 but
    max_dd=-0.05 scores better than Sharpe=2.0 with max_dd=-0.10.

## Rules

1. **One change at a time.** Make a single, testable modification per
   experiment. If you change the model, keep features the same. If you
   change features, keep the model the same.

2. **State your hypothesis.** Fill in the `hypothesis` field explaining
   *why* you expect this change to improve the composite score.

3. **All else equal, simpler is better.** Fewer features, simpler models,
   and lower-dimensional hyperparameter spaces.

4. **Respect constraints:** train_size >= 126, test_size >= 21, embargo >= 3.

5. **Learn from failures.** You will see past experiment results. Do not
   repeat configurations that already failed. If a model family consistently
   underperforms, move on.

6. **Diversify exploration.** Don't fixate on one model family. Try at least
   3 different families before deep-tuning one.

7. **Watch for overfitting signals:**
   - Large gap between training and validation metrics
   - High Sharpe but high max drawdown
   - Unstable performance across folds (check fold-level metrics)

## Response Format

Return ONLY a valid JSON object with the same structure as the input config.
Do not include markdown code fences, explanations, or anything else outside
the JSON object.
