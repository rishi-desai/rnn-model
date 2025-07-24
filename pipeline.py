"""
✔ CSV → 30-bar windows → next-bar return labels
✔ 2-layer LSTM → %-return forecast
✔ Model + metadata auto-saved for plug-and-play reuse
✔ All tunables clearly tagged with WHY_THIS_MATTERS notes
"""

# ── Standard Library
import os, pickle, argparse
from typing import List, Union

# ── Third-Party
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import yfinance as yf


# ===========================================================================
# 1. GLOBAL “KNOBS” (Adjust here ➜ code self-updates everywhere else)
# ===========================================================================

# ── DATA WINDOWING
WINDOW = 30          # 👉 How many bars of history feed the model.
                     #    • Bigger = more context but slower + risk of stale info.
                     #    • Smaller = snappier, reacts faster but may miss patterns.
                     #    • If you adjust this value, ensure when you infer you provide the same window size.

VAL_SPLIT = 0.10     # 👉 Portion of data held out for validation (10% by default).
                     #    • Raise to 0.2 if you want stricter overfitting checks.
                     #    • Lower to 0.05 to squeeze out more training data.

# ── NETWORK & TRAINING
LSTM_UNITS_1 = 32    # 👉 Size of first LSTM layer.
LSTM_UNITS_2 = 16    # 👉 Size of second LSTM layer (set 0 to disable and go 1-layer).
LEARNING_RATE = 1e-3 # 👉 Bigger = learns faster but may wobble; smaller = steadier.
EPOCHS        = 30   # 👉 Training passes through data.  ↑ = potentially better fit.
BATCH_SIZE    = 64   # 👉 Samples per gradient step.  Power-of-2 values are GPU-friendly.
EARLY_PATIENCE= 5    # 👉 How many “bad” validation epochs before training auto-stops.

# ── FILE LOCATIONS
MODEL_DIR  = "model_artifact"           # All outputs land here.
MODEL_PATH = os.path.join(MODEL_DIR, "lstm.h5")
META_PATH  = os.path.join(MODEL_DIR,  "meta.pkl")

# ── RANDOMNESS CONTROL
SEED = 42             # 👉 Fix for reproducibility.  Change to reshuffle weight init etc.
tf.random.set_seed(SEED)
np.random.seed(SEED)


# ===========================================================================
# 2. DATA ENGINEERING
# ===========================================================================

def load_prices(csv_path: str) -> pd.Series:
    """
    Reads <timestamp, adj_close> CSV.

    csv_path : str  - e.g. 'prices.csv'
    returns   : pd.Series of floats indexed by timestamp
    """
    df = pd.read_csv(csv_path, parse_dates=[0], index_col=0)

    # Safety net: insist on ONE price column.
    if df.shape[1] != 1:
        raise ValueError("CSV must contain exactly one numeric column (Adj Close).")

    # Enforce chronological order & drop NaNs.
    return df.iloc[:, 0].sort_index().dropna()


def fetch_prices(ticker: str, period: str = "max") -> pd.Series:
    """
    Downloads historical adjusted close prices for a ticker via yfinance.
    """
    # Ensure we retrieve raw Adj Close column
    df = yf.download(ticker, period=period, auto_adjust=False)
    # Determine which column to use for adjusted prices
    if "Adj Close" in df.columns:
        price_col = "Adj Close"
    elif "Close" in df.columns:
        price_col = "Close"
    else:
        raise ValueError(f"No 'Adj Close' or 'Close' data for ticker: {ticker}")
    series = df[price_col].dropna()
    series.index = pd.to_datetime(series.index)
    return series.sort_index()


def make_windows(prices: pd.Series, window: int = WINDOW):
    """
    Converts price series into overlapping windows + next-bar labels.

    returns
    -------
    X : (N, window, 1) float32  - normalized %-shape windows
    y : (N, 1)        float32  - next-bar return in %
    """
    p = prices.values
    X, y = [], []

    for i in range(len(p) - window):
        segment = p[i : i + window + 1]                 # last value used for label
        base    = segment[0]

        # ── HOW WE SCALE
        # Divide by first price → subtract 1 ➜ every window starts at 0 (0 %=baseline)
        normed_segment = (segment[:-1] / base) - 1.0

        # ── LABEL: % move from last bar in window to *next* bar
        ret_next = (segment[-1] / segment[-2] - 1.0) * 100.0

        X.append(normed_segment.reshape(-1, 1))
        y.append([ret_next])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ===========================================================================
# 3. MODEL ARCHITECTURE
# ===========================================================================

def build_model(input_len: int = WINDOW) -> tf.keras.Model:
    """
    Two-layer (or one-layer) LSTM followed by linear regression head.
    """
    model = models.Sequential(
        [
            layers.Input(shape=(input_len, 1)),
            layers.LSTM(LSTM_UNITS_1, return_sequences=(LSTM_UNITS_2 > 0)),
            # Optional second layer
            *( [layers.LSTM(LSTM_UNITS_2)] if LSTM_UNITS_2 > 0 else [] ),
            layers.Dense(1, activation="linear"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


# ===========================================================================
# 4. TRAINING LOOP
# ===========================================================================

def train(csv_path: str = None,
          ticker: str = None,
          period: str = "max",
          epochs: int = EPOCHS,
          batch: int  = BATCH_SIZE):
    """
    High-level one-call trainer.
    """
    if ticker:
        prices = fetch_prices(ticker, period)
    else:
        prices = load_prices(csv_path)
    X, y   = make_windows(prices)

    # ── Train / Val split
    split        = int(len(X) * (1 - VAL_SPLIT))
    X_train, X_v = X[:split], X[split:]
    y_train, y_v = y[:split], y[split:]

    model = build_model()

    cb_early = callbacks.EarlyStopping(
        monitor="val_mae",
        patience=EARLY_PATIENCE,
        restore_best_weights=True,
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_v, y_v),
        epochs=epochs,
        batch_size=batch,
        callbacks=[cb_early],
        verbose=2,
    )

    # ── Persist everything
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH, include_optimizer=False)

    meta = dict(
        window           = WINDOW,
        norm_scheme      = "divide_by_first_minus_one",
        train_rows       = len(X_train),
        val_rows         = len(X_v),
        lstm_units_1     = LSTM_UNITS_1,
        lstm_units_2     = LSTM_UNITS_2,
        learning_rate    = LEARNING_RATE,
        seed             = SEED,
    )
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)

    print(f"✅ Training done. Model + meta dropped  into: {MODEL_DIR}")


# ===========================================================================
# 5. INFERENCE WRAPPER
# ===========================================================================

class ReturnForecaster:
    """
    One-liner inference façade:

        f = ReturnForecaster()
        pct = f.forecast(latest_30_prices)
    """

    def __init__(self, dir: str = MODEL_DIR):
        # Auto-load artifacts
        with open(os.path.join(dir, "meta.pkl"), "rb") as f:
            self.meta = pickle.load(f)
        self.model = tf.keras.models.load_model(os.path.join(dir, "lstm.h5"), compile=False)

    def _pre(self, latest: Union[List[float], np.ndarray]):
        if len(latest) != self.meta["window"]:
            raise ValueError(f"Need {self.meta['window']} prices, got {len(latest)}.")
        latest = np.asarray(latest, dtype=np.float32)
        return ((latest / latest[0]) - 1.0).reshape(1, -1, 1)

    def forecast(self, latest_30_prices):
        """
        ↳ Returns float: predicted next-bar % move (e.g. +1.23).
        """
        X = self._pre(latest_30_prices)
        return float(self.model.predict(X, verbose=0)[0, 0])


# ===========================================================================
# 6. COMMAND-LINE FRONT DOOR – “zero-dev-ops” mode
# ===========================================================================

def _cli():
    """
    python pipeline.py train prices.csv  --epochs 40
    python pipeline.py infer 123.4 ... 132.7
    """
    p = argparse.ArgumentParser(description="Train or infer with the LSTM forecaster.")
    sub = p.add_subparsers(dest="cmd", required=True)

    # TRAIN
    t = sub.add_parser("train")
    group = t.add_mutually_exclusive_group(required=True)
    group.add_argument("--csv", help="CSV file with <timestamp, adj_close>")
    group.add_argument("--ticker", help="Ticker symbol for automatic data fetch")
    t.add_argument("--period", default="max", help="Period for ticker data (e.g., 1y, 5y, max)")
    t.add_argument("--epochs", type=int, default=EPOCHS,
                   help=f"How many passes over data (default {EPOCHS}).")
    t.add_argument("--batch",  type=int, default=BATCH_SIZE,
                   help=f"Mini-batch size (default {BATCH_SIZE}).")

    # INFER
    i = sub.add_parser("infer")
    i.add_argument("prices", nargs=WINDOW, type=float,
                   help=f"Exactly {WINDOW} latest adjusted closes, oldest → newest.")
    i.add_argument("--ticker", help="Ticker symbol for automatic inference")

    args = p.parse_args()

    if args.cmd == "train":
        train(csv_path=args.csv, ticker=args.ticker, period=args.period, epochs=args.epochs, batch=args.batch)
    else:
        if args.ticker:
            series = fetch_prices(args.ticker, period="max")
            latest = list(series.values[-WINDOW:])
            pct = ReturnForecaster().forecast(latest)
            print(f"RAW: {pct:+.6f} %")
            print(f"Forecasted next-bar return: {pct:+.2f} %")
        else:
            pct = ReturnForecaster().forecast(args.prices)
            print(f"RAW: {pct:+.6f} %")
            print(f"Forecasted next-bar return: {pct:+.2f} %")


# Boot if run as script
if __name__ == "__main__":
    _cli()