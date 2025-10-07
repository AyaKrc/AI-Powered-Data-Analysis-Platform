import os
from typing import List, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool


# ---- Tool declarations (interfaces for the LLM) ----

@tool("summarize_dataset")
def tool_summarize_dataset() -> str:
    """Summarize the loaded dataframe (rows, cols, nulls, dtypes)."""


@tool("correlation_heatmap")
def tool_correlation_heatmap(cols: Optional[List[str]] = None) -> str:
    """Create a correlation heatmap for numeric columns. Optional: cols list."""


@tool("groupby_aggregate")
def tool_groupby_aggregate(by: str, agg_col: str, how: str = "sum", topk: int = 10) -> str:
    """Groupby agg and bar chart. Args: by (group col), agg_col (numeric), how, topk."""


@tool("histogram")
def tool_histogram(col: str, bins: int = 30) -> str:
    """Plot a histogram for a numeric column. Args: col, bins."""


@tool("tiny_classifier")
def tool_tiny_classifier(target: str, features: List[str]) -> str:
    """Train a tiny logistic regression and report accuracy. Args: target, features."""


# Existing plot tools
@tool("pie_chart")
def tool_pie_chart(
    by: str,
    value: Optional[str] = None,
    how: str = "count",
    topk: int = 10,
    donut: bool = True,
) -> str:
    """Pie/Donut chart of shares by category. If value is None, uses counts; else aggregates value with how."""


@tool("line_chart")
def tool_line_chart(
    x: str,
    y: Optional[str] = None,
    how: str = "count",
    topk: Optional[int] = None,
) -> str:
    """Line chart over x. If y is None: counts by x. Else aggregate y by x with how."""


@tool("scatter_plot")
def tool_scatter_plot(
    x: str,
    y: str,
    color: Optional[str] = None,
    max_points: int = 20000,
) -> str:
    """Scatter plot of numeric y vs x. Optional color category. May downsample for speed."""


@tool("box_plot")
def tool_box_plot(
    x: str,
    y: str,
    topk: int = 10,
) -> str:
    """Box plot of numeric y by categorical x (keeps top-k categories by count)."""


@tool("pivot_heatmap")
def tool_pivot_heatmap(
    rows: str,
    cols: str,
    value: Optional[str] = None,
    how: str = "count",
    topk_rows: int = 12,
    topk_cols: int = 12,
) -> str:
    """2D aggregation heatmap (rows × cols). If value is None uses counts; else aggregates value with how."""


# NEW tool declarations
@tool("time_series_decomposition")
def tool_time_series_decomposition(
    date_col: str,
    value_col: str,
    period: int = 12
) -> str:
    """Decompose time series into trend, seasonal, and residual. Args: date_col, value_col, period."""


@tool("outlier_detection")
def tool_outlier_detection(
    col: str,
    method: str = "iqr",
    threshold: float = 1.5
) -> str:
    """Detect outliers using IQR or z-score. Args: col, method (iqr/zscore), threshold."""


@tool("distribution_comparison")
def tool_distribution_comparison(
    col: str,
    by: str,
    topk: int = 5
) -> str:
    """Compare distributions of numeric col across categories. Shows violin plots."""


@tool("feature_importance")
def tool_feature_importance(
    target: str,
    features: Optional[List[str]] = None,
    max_features: int = 10
) -> str:
    """Calculate feature importance using Random Forest. Shows most predictive features."""


@tool("data_quality_report")
def tool_data_quality_report() -> str:
    """Generate comprehensive data quality report with recommendations."""


TOOLS = [
    tool_summarize_dataset,
    tool_correlation_heatmap,
    tool_groupby_aggregate,
    tool_histogram,
    tool_tiny_classifier,
    tool_pie_chart,
    tool_line_chart,
    tool_scatter_plot,
    tool_box_plot,
    tool_pivot_heatmap,
    tool_time_series_decomposition,
    tool_outlier_detection,
    tool_distribution_comparison,
    tool_feature_importance,
    tool_data_quality_report,
]


def load_system_prompt(prompt_file: str = "prompts/system_analyst.md") -> str:
    """Load system prompt from file."""
    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        # Return default prompt if file not found
        return """You are a careful data-analyst copilot for a Streamlit app.

Context:
- A pandas DataFrame named `df` is already loaded from the user's CSV.
- You have access to tools that operate on `df`. Use the fewest tools needed.
- Work in steps: (1) check schema, (2) pick columns, (3) compute, (4) summarize.
- Keep answers short and business-oriented (3–5 insights max).
- If a requested column doesn't exist, suggest close valid alternatives from the schema.
- Prefer clear plots; avoid over-plotting.

When you need computation/plots, CALL a tool with arguments only (no code)."""


def make_chain(model_id: str | None = None):
    """Create main analysis chain."""
    model_id = model_id or os.getenv("MODEL_ID", "gemini-2.0-flash-exp")
    llm = ChatGoogleGenerativeAI(
        model=model_id,
        temperature=0,  # deterministic for demos
    ).bind_tools(TOOLS)

    SYSTEM = load_system_prompt()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM),
            ("human", "User question: {question}\n\nSchema: {schema}"),
        ]
    )
    return prompt | llm


def make_dashboard_chain(model_id: str | None = None):
    """Create chain specifically for auto-dashboard generation."""
    model_id = model_id or os.getenv("MODEL_ID", "gemini-2.0-flash-exp")
    llm = ChatGoogleGenerativeAI(
        model=model_id,
        temperature=0.3,  # Some creativity for dashboard generation
    ).bind_tools(TOOLS)

    DASHBOARD_SYSTEM = """You are an expert data analyst that creates insightful dashboards.

Your task: Analyze the dataset schema and sample data to propose 4-8 visualizations that would create a comprehensive dashboard.

Guidelines for dashboard creation:
1. **Start with Overview**: Always begin with a data summary or key metrics
2. **Distributions**: Include 1-2 distribution plots for important numeric columns
3. **Relationships**: Add correlation or scatter plots to show relationships
4. **Comparisons**: Use groupby/aggregate charts to compare categories
5. **Time Series**: If date columns exist, include time-based analysis
6. **Data Quality**: Consider including outlier detection or quality checks

Selection criteria:
- Choose visualizations that tell a story about the data
- Prioritize columns with business meaning (revenue, sales, dates, categories)
- Mix different chart types for visual variety
- Focus on actionable insights
- Avoid redundant visualizations

For each visualization, select:
- The most appropriate tool
- Meaningful column combinations
- Reasonable parameters (topk=10, bins=30, etc.)

IMPORTANT: 
- Return 4-8 tool calls that create a balanced dashboard
- Each tool call should provide unique insights
- Order visualizations from overview to specific details
- Do NOT include any explanatory text, just make the tool calls"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", DASHBOARD_SYSTEM),
            ("human", """Dataset Overview:
Shape: {shape}
Schema: {schema}

Sample Data (first 10 rows):
{sample}

Generate a comprehensive dashboard with 4-8 visualizations."""),
        ]
    )
    return prompt | llm