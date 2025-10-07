import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy import stats
from difflib import get_close_matches
import warnings
warnings.filterwarnings('ignore')

# ---------- Helpers

def _closest(existing, name, n=3):
    return get_close_matches(name, existing, n=n, cutoff=0.3)

def _as_int(n, default):
    """Safely coerce model-provided numbers (often floats) to positive ints."""
    try:
        return max(1, int(float(n)))
    except Exception:
        return default

def _as_bool(x, default=False):
    """Coerce truthy strings/numbers to bool."""
    if isinstance(x, bool):
        return x
    try:
        s = str(x).strip().lower()
        if s in {"true", "1", "yes", "y", "t"}:
            return True
        if s in {"false", "0", "no", "n", "f"}:
            return False
    except Exception:
        pass
    return default

# Map common human terms to pandas-compatible aggregations
_AGG_ALIASES = {
    # means
    "avg": "mean", "average": "mean", "mean": "mean",
    # sums
    "sum": "sum", "total": "sum",
    # counts
    "count": "count", "size": "count", "n": "count",
    # distinct counts
    "nunique": "nunique", "distinct": "nunique",
    "count_distinct": "nunique", "unique_count": "nunique", "unique": "nunique",
    # order stats
    "median": "median", "p50": "median",
    "min": "min", "max": "max",
    # dispersion
    "std": "std", "stdev": "std",
    "variance": "var", "var": "var",
    # quantiles
    "q1": "quantile_0.25", "q3": "quantile_0.75",
    "p90": "quantile_0.9", "p95": "quantile_0.95", "p99": "quantile_0.99",
}

def _resolve_agg(how: str):
    """
    Normalize a human-friendly word like 'avg' or 'count distinct' into a pandas agg.
    Returns either a string (e.g., 'mean') or a callable (for quantiles).
    """
    if not how:
        return "sum"
    h = how.strip().lower().replace("-", "_").replace(" ", "_")
    h = _AGG_ALIASES.get(h, h)
    if h.startswith("quantile_"):
        try:
            q = float(h.split("_", 1)[1])
        except Exception:
            q = 0.5
        return lambda s: s.quantile(q)
    return h

def _coerce_numeric(series: pd.Series) -> pd.Series:
    """
    Try to coerce a possibly messy numeric column to float:
    strips $, %, commas, spaces, and other non-numeric characters.
    """
    if np.issubdtype(series.dropna().dtype, np.number):
        return series
    coerced = pd.to_numeric(
        series.astype(str).str.replace(r"[^0-9eE\.\-+]", "", regex=True),
        errors="coerce",
    )
    return coerced

# ---------- Core Tools

def summarize_dataset(df: pd.DataFrame) -> dict:
    """Summarize the loaded dataframe (rows, cols, nulls, dtypes)."""
    summary = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "null_counts": df.isna().sum().to_dict(),
        "dtypes": {k: str(v) for k, v in df.dtypes.items()},
    }
    return {
        "kind": "summary",
        "summary": summary,
        "insights": [
            f"Rows: {summary['rows']}, Cols: {summary['cols']}.",
            f"Columns with nulls: {', '.join([k for k, v in summary['null_counts'].items() if v > 0]) or 'none'}.",
        ],
    }

def correlation_heatmap(df: pd.DataFrame, cols=None) -> dict:
    num = df.select_dtypes(include=[np.number])
    if cols:
        missing = [c for c in cols if c not in num.columns]
        if missing:
            return {
                "kind": "error",
                "error": (
                    f"Missing columns for correlation: {missing}. "
                    f"Closest: {{ {', '.join([f'{m}: {_closest(num.columns, m)}' for m in missing])} }}"
                ),
            }
        num = num[cols]
    if num.shape[1] < 2:
        return {"kind": "error", "error": "Need at least 2 numeric columns for a correlation heatmap."}

    corr = num.corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation heatmap")

    insights = []
    try:
        cstack = corr.where(~np.eye(corr.shape[0], dtype=bool)).stack().sort_values(ascending=False)
        top = cstack.head(3)
        for (a, b), v in top.items():
            insights.append(f"High correlation: {a} ~ {b} = {v:.2f}")
    except Exception:
        pass

    return {"kind": "plot", "fig": fig, "insights": insights}

def groupby_aggregate(df: pd.DataFrame, by: str, agg_col: str, how: str = "sum", topk: int = 10) -> dict:
    cols = df.columns
    errs = []
    if by not in cols:
        errs.append(f"`{by}` not found. Close: {_closest(cols, by)}")
    if agg_col not in cols:
        errs.append(f"`{agg_col}` not found. Close: {_closest(cols, agg_col)}")
    if errs:
        return {"kind": "error", "error": " | ".join(errs)}

    # Ensure aggregation column is numeric (coerce if needed)
    series = df[agg_col]
    if not np.issubdtype(series.dropna().dtype, np.number):
        coerced = _coerce_numeric(series)
        if coerced.notna().any():
            df = df.copy()
            df[agg_col] = coerced
        if not np.issubdtype(df[agg_col].dropna().dtype, np.number):
            return {"kind": "error", "error": f"`{agg_col}` must be numeric for aggregation."}

    aggfunc = _resolve_agg(how)

    try:
        table = df.groupby(by, dropna=False)[agg_col].agg(aggfunc).reset_index()
    except Exception as e:
        return {"kind": "error", "error": f"Aggregation failed: {e}"}

    if table.empty:
        empty = pd.DataFrame({by: [], agg_col: []})
        fig = px.bar(empty, x=by, y=agg_col, title=f"{how} of {agg_col} by {by} (Top {topk})")
        return {"kind": "plot+text", "fig": fig, "insight": "No data."}

    topk = _as_int(topk, 10)
    table = table.sort_values(agg_col, ascending=False).head(topk)
    fig = px.bar(table, x=by, y=agg_col, title=f"{how} of {agg_col} by {by} (Top {topk})")
    insight = f"Top group: {table.iloc[0][by]} with {table.iloc[0][agg_col]:,.2f}"
    return {"kind": "plot+text", "fig": fig, "insight": insight}

def histogram(df: pd.DataFrame, col: str, bins: int = 30) -> dict:
    bins = _as_int(bins, 30)
    if col not in df.columns:
        return {"kind": "error", "error": f"`{col}` not found. Close: {_closest(df.columns, col)}"}

    series = df[col]
    if not np.issubdtype(series.dropna().dtype, np.number):
        series = _coerce_numeric(series)
        if series.notna().any():
            df = df.copy()
            df[col] = series

    if not np.issubdtype(df[col].dropna().dtype, np.number):
        return {"kind": "error", "error": f"`{col}` is not numeric."}

    fig = px.histogram(df, x=col, nbins=bins, title=f"Histogram of {col}")
    return {"kind": "plot", "fig": fig, "insights": []}

def tiny_classifier(df: pd.DataFrame, target: str, features: list) -> dict:
    for c in [target] + features:
        if c not in df.columns:
            return {"kind": "error", "error": f"`{c}` not found. Close: {_closest(df.columns, c)}"}

    data = df.dropna(subset=[target] + features).copy()

    # numeric-only for this quick demo
    X = data[features].select_dtypes(include=[np.number])
    if X.shape[1] != len(features):
        return {"kind": "error", "error": "All features must be numeric for this quick demo classifier."}

    y = data[target]
    try:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=7, stratify=None)
        model = LogisticRegression(max_iter=1000)
        model.fit(Xtr, ytr)
        acc = model.score(Xte, yte)
    except Exception as e:
        return {"kind": "error", "error": f"Training failed: {e}"}

    return {"kind": "metric", "metric": f"LogReg test accuracy: {acc:.3f}"}

# ---------- Plot Tools

def pie_chart(
    df: pd.DataFrame,
    by: str,
    value: str | None = None,
    how: str = "count",
    topk: int = 10,
    donut: bool = True,
) -> dict:
    """Make a pie/donut chart of shares by category."""
    if by not in df.columns:
        return {"kind": "error", "error": f"`{by}` not found. Close: {_closest(df.columns, by)}"}

    topk = _as_int(topk, 10)
    donut = _as_bool(donut, True)

    # Build the table
    if value is None:
        table = df.groupby(by, dropna=False).size().reset_index(name="count")
        values_col = "count"
        title = f"Share of {by} (count, Top {topk})"
    else:
        if value not in df.columns:
            return {"kind": "error", "error": f"`{value}` not found. Close: {_closest(df.columns, value)}"}

        series = df[value]
        if not np.issubdtype(series.dropna().dtype, np.number):
            series = _coerce_numeric(series)
            if series.notna().any():
                df = df.copy()
                df[value] = series
        if not np.issubdtype(df[value].dropna().dtype, np.number):
            return {"kind": "error", "error": f"`{value}` must be numeric for aggregation."}

        aggfunc = _resolve_agg(how)
        try:
            table = df.groupby(by, dropna=False)[value].agg(aggfunc).reset_index()
        except Exception as e:
            return {"kind": "error", "error": f"Aggregation failed: {e}"}
        values_col = value
        title = f"Share of {by} by {how}({value}) (Top {topk})"

    # Sort & keep topk
    table = table.sort_values(values_col, ascending=False)
    idx = int(topk)  # enforce int for iloc to avoid float indexer errors
    if len(table) > idx:
        top = table.head(idx)
        other_val = table[values_col].iloc[idx:].sum()
        if float(other_val) > 0:  # avoid a zero-value "Other" slice
            other_row = {by: "Other", values_col: other_val}
            table = pd.concat([top, pd.DataFrame([other_row])], ignore_index=True)

    if table.empty or table[values_col].sum() == 0:
        return {"kind": "error", "error": "Nothing to plot (all zeros or empty result)."}

    # Plot
    fig = px.pie(
        table,
        names=by,
        values=values_col,
        title=title,
        hole=0.4 if donut else 0,
    )
    fig.update_traces(textinfo="percent+label")

    # Insight
    total = float(table[values_col].sum())
    top_row = table.iloc[0]
    insight = f"Top slice: {top_row[by]} at {top_row[values_col] / total:.1%}."

    return {"kind": "plot+text", "fig": fig, "insight": insight}

def line_chart(
    df: pd.DataFrame,
    x: str,
    y: str | None = None,
    how: str = "count",
    topk: int | None = None,
) -> dict:
    """Line chart over x."""
    if x not in df.columns:
        return {"kind": "error", "error": f"`{x}` not found. Close: {_closest(df.columns, x)}"}

    # Prepare x (coerce to numeric if possible to sort properly)
    x_series = df[x]
    x_is_num = np.issubdtype(x_series.dropna().dtype, np.number)
    if not x_is_num:
        maybe_num = _coerce_numeric(x_series)
        if maybe_num.notna().any() and maybe_num.count() >= x_series.count() * 0.9:
            df = df.copy()
            df[x] = maybe_num
            x_is_num = True

    if y is None:
        table = df.groupby(x, dropna=False).size().reset_index(name="count")
        ycol = "count"
        title = f"Count by {x}"
    else:
        if y not in df.columns:
            return {"kind": "error", "error": f"`{y}` not found. Close: {_closest(df.columns, y)}"}
        series = df[y]
        if not np.issubdtype(series.dropna().dtype, np.number):
            series = _coerce_numeric(series)
            if series.notna().any():
                df = df.copy()
                df[y] = series
        if not np.issubdtype(df[y].dropna().dtype, np.number):
            return {"kind": "error", "error": f"`{y}` must be numeric for aggregation."}
        aggfunc = _resolve_agg(how)
        try:
            table = df.groupby(x, dropna=False)[y].agg(aggfunc).reset_index()
        except Exception as e:
            return {"kind": "error", "error": f"Aggregation failed: {e}"}
        ycol = y
        title = f"{how} of {y} by {x}"

    # Sort x ascending if numeric, else leave natural/group order
    if x_is_num:
        table = table.sort_values(x)

    if topk:
        table = table.head(_as_int(topk, len(table)))

    fig = px.line(table, x=x, y=ycol, markers=True, title=title)

    # Simple trend insight
    try:
        first, last = table[ycol].iloc[0], table[ycol].iloc[-1]
        direction = "up" if last > first else ("down" if last < first else "flat")
        insight = f"Trend is {direction}: {first:,.2f} → {last:,.2f}."
    except Exception:
        insight = None

    return {"kind": "plot+text", "fig": fig, "insight": insight or ""}

def scatter_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str | None = None,
    max_points: int = 20000,
) -> dict:
    """Scatter plot of numeric x vs numeric y. Optionally color by a categorical column."""
    for c in [x, y]:
        if c not in df.columns:
            return {"kind": "error", "error": f"`{c}` not found. Close: {_closest(df.columns, c)}"}

    max_points = _as_int(max_points, 20000)

    # Coerce numerics
    X = _coerce_numeric(df[x])
    Y = _coerce_numeric(df[y])
    data = pd.DataFrame({x: X, y: Y})
    if color and color in df.columns:
        data[color] = df[color]
    data = data.dropna(subset=[x, y])

    if data.empty:
        return {"kind": "error", "error": "No numeric data after coercion."}

    if len(data) > max_points:
        data = data.sample(max_points, random_state=7)

    fig = px.scatter(data, x=x, y=y, color=color, title=f"{y} vs {x}")

    # Correlation insight
    try:
        corr = np.corrcoef(data[x], data[y])[0, 1]
        insight = f"Pearson correlation: {corr:.2f}."
    except Exception:
        insight = ""

    return {"kind": "plot+text", "fig": fig, "insight": insight}

def box_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    topk: int = 10,
) -> dict:
    """Box plot of numeric y by categorical x, keeping top-k categories by count."""
    for c in [x, y]:
        if c not in df.columns:
            return {"kind": "error", "error": f"`{c}` not found. Close: {_closest(df.columns, c)}"}

    topk = _as_int(topk, 10)

    # Ensure y numeric
    y_series = _coerce_numeric(df[y])
    if not np.issubdtype(y_series.dropna().dtype, np.number):
        return {"kind": "error", "error": f"`{y}` must be numeric."}
    dfx = df.copy()
    dfx[y] = y_series

    # Top-k categories by count
    counts = dfx[x].value_counts(dropna=False).head(topk).index
    dfx = dfx[dfx[x].isin(counts)]

    if dfx.empty:
        return {"kind": "error", "error": "Nothing to plot after filtering."}

    fig = px.box(dfx, x=x, y=y, points=False, title=f"Distribution of {y} by {x} (Top {topk})")

    # Insight: which category has the highest median?
    med = dfx.groupby(x)[y].median().sort_values(ascending=False)
    top_cat, top_val = med.index[0], med.iloc[0]
    insight = f"Highest median {y}: {top_cat} at {top_val:,.2f}."

    return {"kind": "plot+text", "fig": fig, "insight": insight}

def pivot_heatmap(
    df: pd.DataFrame,
    rows: str,
    cols: str,
    value: str | None = None,
    how: str = "count",
    topk_rows: int = 12,
    topk_cols: int = 12,
) -> dict:
    """2D aggregation heatmap (rows × cols)."""
    for c in [rows, cols]:
        if c not in df.columns:
            return {"kind": "error", "error": f"`{c}` not found. Close: {_closest(df.columns, c)}"}

    topk_rows = _as_int(topk_rows, 12)
    topk_cols = _as_int(topk_cols, 12)

    dfx = df.copy()

    if value is None:
        # Use a stable column for counting (any existing column will do)
        count_col = dfx.columns[0]
        pt = dfx.pivot_table(index=rows, columns=cols, values=count_col, aggfunc="count", fill_value=0)
        title = f"Count heatmap: {rows} × {cols}"
    else:
        if value not in dfx.columns:
            return {"kind": "error", "error": f"`{value}` not found. Close: {_closest(dfx.columns, value)}"}
        val_series = _coerce_numeric(dfx[value])
        if not np.issubdtype(val_series.dropna().dtype, np.number):
            return {"kind": "error", "error": f"`{value}` must be numeric for aggregation."}
        dfx[value] = val_series
        aggfunc = _resolve_agg(how)
        pt = dfx.pivot_table(index=rows, columns=cols, values=value, aggfunc=aggfunc, fill_value=0)
        title = f"{how}({value}) heatmap: {rows} × {cols}"

    # keep top rows/cols by totals
    row_tot = pt.sum(axis=1).sort_values(ascending=False).head(topk_rows).index
    col_tot = pt.sum(axis=0).sort_values(ascending=False).head(topk_cols).index
    pt = pt.loc[row_tot, col_tot]

    if pt.empty:
        return {"kind": "error", "error": "Nothing to plot after limiting rows/cols."}

    fig = px.imshow(pt, aspect="auto", title=title, labels=dict(x=cols, y=rows, color=value or "count"))

    # Insight: top cell
    try:
        max_idx = np.unravel_index(np.argmax(pt.values), pt.shape)
        r, c = pt.index[max_idx[0]], pt.columns[max_idx[1]]
        v = pt.values[max_idx]
        insight = f"Strongest cell: {rows}={r}, {cols}={c} → {v:,.2f}"
    except Exception:
        insight = ""

    return {"kind": "plot+text", "fig": fig, "insight": insight}

# ---------- NEW Analysis Capabilities (theme-friendly)

def time_series_decomposition(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    period: int = 12
) -> dict:
    """Decompose time series into trend, seasonal, and residual components."""
    if date_col not in df.columns:
        return {"kind": "error", "error": f"`{date_col}` not found. Close: {_closest(df.columns, date_col)}"}
    if value_col not in df.columns:
        return {"kind": "error", "error": f"`{value_col}` not found. Close: {_closest(df.columns, value_col)}"}
    
    try:
        # Prepare data
        dfx = df[[date_col, value_col]].copy()
        dfx[date_col] = pd.to_datetime(dfx[date_col], errors='coerce')
        dfx = dfx.dropna()
        dfx = dfx.sort_values(date_col)
        dfx = dfx.groupby(date_col)[value_col].mean().reset_index()

        if len(dfx) < period * 2:
            return {"kind": "error", "error": f"Need at least {period * 2} data points for decomposition"}

        # Simple moving average for trend
        dfx['trend'] = dfx[value_col].rolling(window=period, center=True).mean()

        # Detrend
        dfx['detrended'] = dfx[value_col] - dfx['trend']

        # Seasonal pattern (average by period position)
        dfx['period_pos'] = np.arange(len(dfx)) % period
        seasonal = dfx.groupby('period_pos')['detrended'].mean().to_dict()
        dfx['seasonal'] = dfx['period_pos'].map(seasonal)

        # Residual
        dfx['residual'] = dfx['detrended'] - dfx['seasonal']

        # Create multi-trace figure WITHOUT hard-coded colors (inherits palette)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dfx[date_col], y=dfx[value_col], name='Original'))
        fig.add_trace(go.Scatter(x=dfx[date_col], y=dfx['trend'], name='Trend', line=dict(width=2)))
        fig.add_trace(go.Scatter(x=dfx[date_col], y=dfx['seasonal'], name='Seasonal'))
        fig.add_trace(go.Scatter(x=dfx[date_col], y=dfx['residual'], name='Residual'))
        fig.update_layout(
            title=f"Time Series Decomposition: {value_col}",
            xaxis_title=date_col,
            yaxis_title="Value",
            hovermode='x unified'
        )

        # Insights (robust to NaNs on edges)
        t = dfx['trend'].dropna()
        if len(t) >= 2:
            if t.iloc[-1] > t.iloc[0]:
                trend_direction = "increasing"
            elif t.iloc[-1] < t.iloc[0]:
                trend_direction = "decreasing"
            else:
                trend_direction = "flat"
        else:
            trend_direction = "unclear (insufficient data)"

        denom = dfx[value_col].std()
        seasonal_strength = (dfx['seasonal'].std() / denom) if denom and denom > 0 else 0.0

        insights = [
            f"Overall trend is {trend_direction}",
            f"Seasonal strength: {seasonal_strength:.2%} of total variation",
            f"Period: {period} time units"
        ]

        return {"kind": "plot", "fig": fig, "insights": insights}
        
    except Exception as e:
        return {"kind": "error", "error": f"Decomposition failed: {str(e)}"}

def outlier_detection(
    df: pd.DataFrame,
    col: str,
    method: str = "iqr",
    threshold: float = 1.5
) -> dict:
    """Detect outliers using IQR or Z-score method."""
    if col not in df.columns:
        return {"kind": "error", "error": f"`{col}` not found. Close: {_closest(df.columns, col)}"}
    
    series = _coerce_numeric(df[col])
    if not np.issubdtype(series.dropna().dtype, np.number):
        return {"kind": "error", "error": f"`{col}` must be numeric for outlier detection."}
    
    clean_data = series.dropna()
    if len(clean_data) < 2:
        return {"kind": "error", "error": f"Not enough data in `{col}` for outlier detection."}
    
    if method.lower() == "iqr":
        Q1 = clean_data.quantile(0.25)
        Q3 = clean_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers = (clean_data < lower_bound) | (clean_data > upper_bound)
    else:  # z-score
        z_scores = np.abs(stats.zscore(clean_data))
        outliers = z_scores > threshold
    
    # Create visualization (no hard-coded colors)
    fig = go.Figure()
    fig.add_trace(go.Box(
        y=clean_data,
        name=col,
        boxpoints='outliers',
    ))
    
    # Add scatter for outliers (keep symbol/size, no color so palette applies)
    outlier_values = clean_data[outliers]
    if len(outlier_values) > 0:
        fig.add_trace(go.Scatter(
            x=[col] * len(outlier_values),
            y=outlier_values,
            mode='markers',
            name='Outliers',
            marker=dict(size=8, symbol='x')
        ))
    
    fig.update_layout(
        title=f"Outlier Detection: {col} ({method.upper()} method)",
        yaxis_title="Value"
    )
    
    n_outliers = int(outliers.sum())
    pct_outliers = n_outliers / len(clean_data) * 100
    
    insights = [
        f"Found {n_outliers} outliers ({pct_outliers:.1f}% of data)",
        f"Method: {method.upper()} with threshold {threshold}",
        f"Data range: [{clean_data.min():.2f}, {clean_data.max():.2f}]"
    ]
    
    if n_outliers > 0:
        insights.append(f"Outlier range: [{outlier_values.min():.2f}, {outlier_values.max():.2f}]")
    
    return {"kind": "plot", "fig": fig, "insights": insights}

def distribution_comparison(
    df: pd.DataFrame,
    col: str,
    by: str,
    topk: int = 5
) -> dict:
    """Compare distributions of a numeric column across categories."""
    if col not in df.columns:
        return {"kind": "error", "error": f"`{col}` not found. Close: {_closest(df.columns, col)}"}
    if by not in df.columns:
        return {"kind": "error", "error": f"`{by}` not found. Close: {_closest(df.columns, by)}"}
    
    # Ensure numeric column
    series = _coerce_numeric(df[col])
    if not np.issubdtype(series.dropna().dtype, np.number):
        return {"kind": "error", "error": f"`{col}` must be numeric."}
    
    dfx = df.copy()
    dfx[col] = series
    
    # Get top k categories
    top_cats = dfx[by].value_counts().head(_as_int(topk, 5)).index
    dfx = dfx[dfx[by].isin(top_cats)]
    
    # Violin plot inherits palette; do NOT force legend here (let app decide)
    fig = px.violin(
        dfx, 
        x=by, 
        y=col,
        color=by,
        box=True,
        points=False,
        title=f"Distribution Comparison: {col} by {by}"
    )
    
    # Statistical insights
    insights = []
    stats_by_cat = dfx.groupby(by)[col].agg(['mean', 'median', 'std'])
    
    highest_mean = stats_by_cat['mean'].idxmax()
    lowest_mean = stats_by_cat['mean'].idxmin()
    highest_var = stats_by_cat['std'].idxmax()
    
    insights.append(f"Highest mean: {highest_mean} ({stats_by_cat.loc[highest_mean, 'mean']:.2f})")
    insights.append(f"Lowest mean: {lowest_mean} ({stats_by_cat.loc[lowest_mean, 'mean']:.2f})")
    insights.append(f"Most variable: {highest_var} (std: {stats_by_cat.loc[highest_var, 'std']:.2f})")
    
    return {"kind": "plot", "fig": fig, "insights": insights}

def feature_importance(
    df: pd.DataFrame,
    target: str,
    features: list = None,
    max_features: int = 10
) -> dict:
    """Calculate feature importance using Random Forest."""
    if target not in df.columns:
        return {"kind": "error", "error": f"`{target}` not found. Close: {_closest(df.columns, target)}"}
    
    # If features not specified, use all numeric columns except target
    if features is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        features = [c for c in numeric_cols if c != target]
    
    if len(features) == 0:
        return {"kind": "error", "error": "No numeric features found for importance analysis"}
    
    # Validate features exist
    missing = [f for f in features if f not in df.columns]
    if missing:
        return {"kind": "error", "error": f"Features not found: {missing}"}
    
    # Prepare data
    data = df[features + [target]].dropna()
    
    if len(data) < 50:
        return {"kind": "error", "error": "Need at least 50 samples for feature importance"}
    
    X = data[features]
    y = data[target]
    
    # Determine if regression or classification
    n_unique = y.nunique()
    is_regression = n_unique > 10 or (np.issubdtype(y.dtype, np.number) and n_unique / len(y) > 0.05)
    
    try:
        if is_regression:
            model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
        
        model.fit(X, y)
        importances = model.feature_importances_
        
        # Create dataframe for plotting
        importance_df = pd.DataFrame({
            'feature': features,
            'importance': importances
        }).sort_values('importance', ascending=True).tail(max_features)
        
        # Create horizontal bar chart
        fig = px.bar(
            importance_df,
            x='importance',
            y='feature',
            orientation='h',
            title=f"Feature Importance for {target}",
            labels={'importance': 'Importance Score', 'feature': 'Feature'}
        )
        
        fig.update_layout(height=max(400, len(importance_df) * 30))
        
        # Insights
        top_feature = importance_df.iloc[-1]
        model_type = "regression" if is_regression else "classification"
        
        insights = [
            f"Most important feature: {top_feature['feature']} ({top_feature['importance']:.3f})",
            f"Model type: Random Forest {model_type}",
            f"Total features analyzed: {len(features)}"
        ]
        
        return {"kind": "plot", "fig": fig, "insights": insights}
        
    except Exception as e:
        return {"kind": "error", "error": f"Feature importance failed: {str(e)}"}

def data_quality_report(df: pd.DataFrame) -> dict:
    """Generate comprehensive data quality report with recommendations."""
    report_lines = ["## Data Quality Report\n"]
    recommendations = []
    
    # Basic stats
    n_rows, n_cols = df.shape
    if n_rows == 0:
        return {
            "kind": "report",
            "report": "**Empty dataset (0 rows)**",
            "recommendations": []
        }
    report_lines.append(f"**Dataset Size:** {n_rows:,} rows × {n_cols} columns\n")
    
    # Missing values analysis
    missing = df.isnull().sum()
    missing_pct = (missing / n_rows * 100).round(2)
    n_missing_cols = (missing > 0).sum()
    
    if n_missing_cols > 0:
        report_lines.append(f"\n### Missing Values")
        report_lines.append(f"- **Columns with missing data:** {n_missing_cols}/{n_cols}")
        
        worst_missing = missing_pct.nlargest(5)
        if len(worst_missing) > 0:
            report_lines.append("- **Top columns by missing %:**")
            for col, pct in worst_missing.items():
                report_lines.append(f"  - {col}: {pct}% missing")
        
        if missing_pct.max() > 50:
            recommendations.append("Consider dropping columns with >50% missing values")
        if missing_pct.max() > 20:
            recommendations.append("Implement imputation strategy for columns with 20-50% missing")
    
    # Duplicates
    n_duplicates = df.duplicated().sum()
    if n_duplicates > 0:
        report_lines.append(f"\n### Duplicates")
        report_lines.append(f"- **Duplicate rows:** {n_duplicates:,} ({n_duplicates/n_rows*100:.1f}%)")
        recommendations.append(f"Remove {n_duplicates:,} duplicate rows")
    
    # Data types analysis
    dtypes_summary = df.dtypes.value_counts()
    report_lines.append(f"\n### Data Types")
    for dtype, count in dtypes_summary.items():
        report_lines.append(f"- **{dtype}:** {count} columns")
    
    # Numeric columns analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        report_lines.append(f"\n### Numeric Columns ({len(numeric_cols)})")
        
        # Check for potential outliers
        outlier_cols = []
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum()
            if outliers > 0:
                outlier_cols.append(f"{col} ({outliers} outliers)")
        
        if outlier_cols:
            report_lines.append(f"- **Columns with outliers:** {len(outlier_cols)}")
            for col_info in outlier_cols[:5]:  # Show top 5
                report_lines.append(f"  - {col_info}")
            recommendations.append("Review and handle outliers in numeric columns")
    
    # Categorical columns analysis
    object_cols = df.select_dtypes(include=['object']).columns
    if len(object_cols) > 0:
        report_lines.append(f"\n### Categorical Columns ({len(object_cols)})")
        
        high_cardinality = []
        for col in object_cols:
            n_unique = df[col].nunique()
            if n_unique > 50:
                high_cardinality.append(f"{col} ({n_unique} unique)")
        
        if high_cardinality:
            report_lines.append(f"- **High cardinality columns:** {len(high_cardinality)}")
            for col_info in high_cardinality[:5]:
                report_lines.append(f"  - {col_info}")
            recommendations.append("Consider encoding or grouping high-cardinality categorical features")
    
    # Memory usage
    memory_usage = df.memory_usage(deep=True).sum() / 1024**2  # Convert to MB
    report_lines.append(f"\n### Memory Usage")
    report_lines.append(f"- **Total:** {memory_usage:.2f} MB")
    report_lines.append(f"- **Per row:** {memory_usage*1024/n_rows:.2f} KB")
    
    if memory_usage > 100:
        recommendations.append("Consider optimizing data types to reduce memory usage")
    
    # Correlation check for numeric columns
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.9:
                    high_corr.append(f"{corr_matrix.columns[i]} & {corr_matrix.columns[j]}: {corr_matrix.iloc[i, j]:.2f}")
        
        if high_corr:
            report_lines.append(f"\n### High Correlations")
            for corr_info in high_corr[:5]:
                report_lines.append(f"- {corr_info}")
            recommendations.append("Review highly correlated features for potential redundancy")
    
    return {
        "kind": "report",
        "report": "\n".join(report_lines),
        "recommendations": recommendations
    }

def preprocess_data(
    df: pd.DataFrame,
    handle_missing: str = "none",
    remove_duplicates: bool = False,
    standardize_text: bool = False,
    detect_datetime: bool = False,
    remove_outliers: bool = False
) -> dict:
    """Preprocess data with various cleaning options."""
    df_processed = df.copy()
    changes = []
    original_shape = df.shape
    
    # Handle missing values
    if handle_missing != "none":
        if handle_missing == "drop_rows":
            df_processed = df_processed.dropna()
            changes.append(f"Dropped {original_shape[0] - df_processed.shape[0]} rows with missing values")
        elif handle_missing == "drop_cols":
            thresh = len(df_processed) * 0.5  # Drop columns with >50% missing
            df_processed = df_processed.dropna(axis=1, thresh=thresh)
            changes.append(f"Dropped {original_shape[1] - df_processed.shape[1]} columns with >50% missing")
        elif handle_missing == "fill_mean":
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].mean())
            changes.append("Filled numeric missing values with mean")
        elif handle_missing == "fill_median":
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].median())
            changes.append("Filled numeric missing values with median")
        elif handle_missing == "fill_mode":
            for col in df_processed.columns:
                if df_processed[col].isnull().any():
                    mode_val = df_processed[col].mode()
                    if len(mode_val) > 0:
                        df_processed[col] = df_processed[col].fillna(mode_val[0])
            changes.append("Filled missing values with mode")
        elif handle_missing == "fill_forward":
            df_processed = df_processed.ffill()
            changes.append("Forward-filled missing values")
        elif handle_missing == "fill_zero":
            df_processed = df_processed.fillna(0)
            changes.append("Filled missing values with zero")
    
    # Remove duplicates
    if remove_duplicates:
        n_before = len(df_processed)
        df_processed = df_processed.drop_duplicates()
        n_removed = n_before - len(df_processed)
        if n_removed > 0:
            changes.append(f"Removed {n_removed} duplicate rows")
    
    # Standardize text columns
    if standardize_text:
        text_cols = df_processed.select_dtypes(include=['object']).columns
        for col in text_cols:
            df_processed[col] = df_processed[col].astype(str).str.strip().str.lower()
            df_processed[col] = df_processed[col].replace('nan', np.nan)
        if len(text_cols) > 0:
            changes.append(f"Standardized {len(text_cols)} text columns")
    
    # Auto-detect datetime columns
    if detect_datetime:
        potential_date_cols = []
        for col in df_processed.select_dtypes(include=['object']).columns:
            try:
                pd.to_datetime(df_processed[col].dropna().head(100))
                df_processed[col] = pd.to_datetime(df_processed[col], errors='coerce')
                potential_date_cols.append(col)
            except Exception:
                pass
        if potential_date_cols:
            changes.append(f"Converted {len(potential_date_cols)} columns to datetime")
    
    # Remove outliers (IQR method)
    if remove_outliers:
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        n_before = len(df_processed)
        for col in numeric_cols:
            Q1 = df_processed[col].quantile(0.25)
            Q3 = df_processed[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df_processed = df_processed[(df_processed[col] >= lower) & (df_processed[col] <= upper)]
        n_removed = n_before - len(df_processed)
        if n_removed > 0:
            changes.append(f"Removed {n_removed} rows with outliers")
    
    # Create summary
    summary = f"Processed {original_shape[0]} rows × {original_shape[1]} cols → {df_processed.shape[0]} rows × {df_processed.shape[1]} cols"
    if changes:
        summary += "\nChanges: " + " | ".join(changes)
    else:
        summary += "\nNo changes applied"
    
    return {
        "df": df_processed,
        "summary": summary,
        "changes": changes
    }
