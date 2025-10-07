import os
import json
import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.io as pio  # NEW

from chains import make_chain, make_dashboard_chain
from tools import (
    summarize_dataset,
    correlation_heatmap,
    groupby_aggregate,
    histogram,
    tiny_classifier,
    pie_chart,
    line_chart,
    scatter_plot,
    box_plot,
    pivot_heatmap,
    time_series_decomposition,   # NEW
    outlier_detection,           # NEW
    distribution_comparison,     # NEW
    feature_importance,          # NEW
    data_quality_report,         # NEW
    preprocess_data,             # NEW preprocessing
)

# ---- Page config (favicon via env, secrets, or local file)
LOCAL_FAVICON = "assets/favicon.png"  # put a PNG here if you want
PAGE_ICON = os.getenv("PAGE_ICON")
try:
    PAGE_ICON = PAGE_ICON or (st.secrets.get("PAGE_ICON") if hasattr(st, "secrets") else None)
except Exception:
    PAGE_ICON = PAGE_ICON  # ignore if secrets missing
if not PAGE_ICON and os.path.exists(LOCAL_FAVICON):
    PAGE_ICON = LOCAL_FAVICON

st.set_page_config(
    page_title="Ask-Your-Data · GenAI Analyst",
    layout="wide",
    page_icon=PAGE_ICON
)
st.title("Ask-Your-Data · GenAI Analyst")

# ---- Config: env OR .streamlit/secrets.toml
def _get_secret(key, default=None):
    val = os.getenv(key)
    if val:
        return val
    try:
        if hasattr(st, "secrets"):
            return st.secrets.get(key, default)
    except Exception:
        pass
    return default

GOOGLE_API_KEY = _get_secret("GOOGLE_API_KEY")
MODEL_ID = _get_secret("MODEL_ID", "gemini-2.5-pro")
MODEL_FALLBACK = _get_secret("MODEL_FALLBACK", "gemini-2.5-flash")

# Re-expose to os.environ in case downstream reads from env
if MODEL_ID:
    os.environ["MODEL_ID"] = MODEL_ID
if MODEL_FALLBACK:
    os.environ["MODEL_FALLBACK"] = MODEL_FALLBACK
if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# ---- Check API key
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY is not set. Add it to .streamlit/secrets.toml or set it as an environment variable.")
    st.stop()

# ---- UI theming options & defaults
QUAL_PALETTES = {
    "Plotly": px.colors.qualitative.Plotly,
    "D3": px.colors.qualitative.D3,
    "Bold": px.colors.qualitative.Bold,
    "Pastel": px.colors.qualitative.Pastel,
    "Prism": px.colors.qualitative.Prism,
    "Set3": px.colors.qualitative.Set3,
    "Safe": px.colors.qualitative.Safe,
}
TEMPLATES = ["plotly_white", "simple_white", "plotly", "ggplot2", "seaborn", "plotly_dark", "presentation"]

DEFAULT_UI = {
    "dash_title": "Intelligent Dashboard",
    "subtitle": "",
    "plotly_template": "plotly_white",
    "palette": "Plotly",
    "columns": 2,             # cards per row
    "show_legends": True,
    "font_size": 14,
    "force_reskin": False,    # aggressively override hard-coded colors
}

# Initialize session state
if "ui_settings" not in st.session_state:
    st.session_state.ui_settings = DEFAULT_UI.copy()

# Initialize dashboard-related session state
if "dashboard_plots" not in st.session_state:
    st.session_state.dashboard_plots = []

if "dashboard_generated" not in st.session_state:
    st.session_state.dashboard_generated = False

# Track custom plots separately if needed for analytics
if "custom_plots_count" not in st.session_state:
    st.session_state.custom_plots_count = 0

# ==== THEME INJECTION HELPERS (key for customization) ====

def _set_px_theme_from_ui():
    """
    Make Plotly Express and Plotly pick up the user-selected template & palette
    BEFORE any figure is created by tools.py.
    """
    s = st.session_state.ui_settings
    # Defaults for plotly express
    px.defaults.template = s["plotly_template"]
    px.defaults.color_discrete_sequence = QUAL_PALETTES.get(s["palette"], px.colors.qualitative.Plotly)
    px.defaults.color_continuous_scale = "Viridis"
    # Set plotly.io global template as well (affects go.Figure too)
    pio.templates.default = s["plotly_template"]

def _strip_explicit_colors(fig):
    """
    Aggressively remove explicit colors on traces so the chosen colorway applies.
    Use when tools.py hard-codes marker/line colors.
    """
    if not getattr(fig, "data", None):
        return fig
    for tr in fig.data:
        # marker color
        if getattr(tr, "marker", None) is not None:
            if hasattr(tr.marker, "color"):
                try:
                    tr.marker.color = None
                except Exception:
                    pass
            if hasattr(tr.marker, "line") and hasattr(tr.marker.line, "color"):
                try:
                    tr.marker.line.color = None
                except Exception:
                    pass
        # line color (scatter/line)
        if getattr(tr, "line", None) is not None and hasattr(tr.line, "color"):
            try:
                tr.line.color = None
            except Exception:
                pass
    return fig

def _apply_theme(fig):
    """Apply global dashboard appearance settings to a Plotly figure."""
    s = st.session_state.ui_settings
    if s.get("force_reskin", False):
        fig = _strip_explicit_colors(fig)
    fig.update_layout(
        template=s["plotly_template"],
        font=dict(size=int(s["font_size"])),
        colorway=QUAL_PALETTES.get(s["palette"], px.colors.qualitative.Plotly),
    )
    if not s["show_legends"]:
        fig.update_layout(showlegend=False)
    return fig

# ---- Helper functions
def generate_auto_dashboard(df, schema):
    """Generate initial dashboard recommendations using AI"""
    with st.spinner("Analyzing your data to suggest visualizations..."):
        try:
            chain = make_dashboard_chain()
            response = chain.invoke({
                "schema": json.dumps(schema),
                "sample": df.head(10).to_json(),
                "shape": f"{df.shape[0]} rows, {df.shape[1]} columns"
            })
            # Extract tool calls from response
            tool_calls = getattr(response, "tool_calls", None)
            if not tool_calls and hasattr(response, "additional_kwargs"):
                tool_calls = response.additional_kwargs.get("tool_calls", [])
            return tool_calls or []
        except Exception as e:
            st.warning(f"Could not generate auto-dashboard: {e}")
            return []

def execute_tool_call(df, call):
    """Execute a single tool call and return the result"""
    # Ensure PX/Plotly defaults reflect current UI **before** any figure creation
    _set_px_theme_from_ui()

    name = call.get("name")
    args = call.get("args", {}) or {}

    # Map tool names to functions
    tool_map = {
        "summarize_dataset": summarize_dataset,
        "correlation_heatmap": correlation_heatmap,
        "groupby_aggregate": groupby_aggregate,
        "histogram": histogram,
        "tiny_classifier": tiny_classifier,
        "pie_chart": pie_chart,
        "line_chart": line_chart,
        "scatter_plot": scatter_plot,
        "box_plot": box_plot,
        "pivot_heatmap": pivot_heatmap,
        "time_series_decomposition": time_series_decomposition,
        "outlier_detection": outlier_detection,
        "distribution_comparison": distribution_comparison,
        "feature_importance": feature_importance,
        "data_quality_report": data_quality_report,
    }

    if name in tool_map:
        return tool_map[name](df, **args)
    else:
        return {"kind": "error", "error": f"Unknown tool {name}"}

def render_result(out, col=None, unique_key=None):
    """Render a tool result in the specified column or main area"""
    context = col if col else st

    kind = out.get("kind", "")
    if kind == "error":
        context.error(out["error"])
    elif kind in ("plot", "plot+text"):
        fig = out["fig"]
        if "ui_settings" in st.session_state:
            fig = _apply_theme(fig)  # theme + (optional) aggressive reskin
        # Use unique key to avoid duplicate element ID errors
        plot_key = f"plot_{unique_key}" if unique_key else None
        context.plotly_chart(fig, use_container_width=True, key=plot_key)
        if "insight" in out and out["insight"]:
            context.success(out["insight"])
        if "insights" in out and out["insights"]:
            context.write("• " + "\n• ".join(out["insights"]))
    elif kind == "summary":
        context.json(out["summary"])
        if out.get("insights"):
            context.write("• " + "\n• ".join(out["insights"]))
    elif kind == "metric":
        context.metric("Result", out["metric"])
    elif kind == "report":
        # For data quality report
        context.markdown(out["report"])
        if out.get("recommendations"):
            context.info("**Recommendations:**\n• " + "\n• ".join(out["recommendations"]))

# ---- Main UI
# Sidebar for data upload, preprocessing, and appearance
with st.sidebar:
    st.header("Data Upload")

    uploaded = st.file_uploader("Upload a CSV", type=["csv"])

    if uploaded and "df_raw" not in st.session_state:
        st.session_state.df_raw = pd.read_csv(uploaded)
        st.session_state.df = st.session_state.df_raw.copy()
        st.session_state.dashboard_generated = False

    # Example loader
    with st.expander("Load example dataset"):
        if st.button("Load marketing_sample.csv"):
            try:
                st.session_state.df_raw = pd.read_csv("examples/marketing_sample.csv")
                st.session_state.df = st.session_state.df_raw.copy()
                st.session_state.dashboard_generated = False
            except FileNotFoundError:
                st.warning("Place a file at examples/marketing_sample.csv first.")

    # Data Preprocessing Options
    if "df_raw" in st.session_state:
        st.header("Data Preprocessing")

        with st.expander("Preprocessing Options", expanded=False):
            handle_missing = st.selectbox(
                "Handle Missing Values",
                ["none", "drop_rows", "drop_cols", "fill_mean", "fill_median", "fill_mode", "fill_forward", "fill_zero"]
            )

            remove_duplicates = st.checkbox("Remove duplicate rows")

            standardize_text = st.checkbox("Standardize text (lowercase, trim)")

            detect_datetime = st.checkbox("Auto-detect datetime columns")

            remove_outliers = st.checkbox("Remove outliers (IQR method)")

            if st.button("Apply Preprocessing"):
                with st.spinner("Preprocessing data..."):
                    options = {
                        "handle_missing": handle_missing,
                        "remove_duplicates": remove_duplicates,
                        "standardize_text": standardize_text,
                        "detect_datetime": detect_datetime,
                        "remove_outliers": remove_outliers,
                    }
                    result = preprocess_data(st.session_state.df_raw, **options)
                    st.session_state.df = result["df"]
                    st.success(f"Preprocessing complete.\n{result['summary']}")
                    st.session_state.dashboard_generated = False

    # ---- Appearance / Theme controls
    st.header("Dashboard Appearance")
    with st.expander("Customize look & feel", expanded=False):
        s = st.session_state.ui_settings
        s["dash_title"] = st.text_input("Dashboard title", s["dash_title"])
        s["subtitle"] = st.text_input("Subtitle (optional)", s["subtitle"])
        s["plotly_template"] = st.selectbox("Plot theme", TEMPLATES, index=TEMPLATES.index(s["plotly_template"]))
        s["palette"] = st.selectbox("Color palette", list(QUAL_PALETTES.keys()),
                                    index=list(QUAL_PALETTES.keys()).index(s["palette"]))
        s["columns"] = st.slider("Cards per row", 1, 3, value=int(s["columns"]))
        s["show_legends"] = st.checkbox("Show legends on charts", value=bool(s["show_legends"]))
        s["font_size"] = st.slider("Base font size", 10, 22, value=int(s["font_size"]))
        s["force_reskin"] = st.checkbox("Force palette override (aggressive)", value=bool(s["force_reskin"]))
        st.session_state.ui_settings = s

# Main area
if "df" in st.session_state:
    df = st.session_state.df

    # Create tabs for different modes
    tab1, tab2, tab3, tab4 = st.tabs(["Auto Dashboard", "Ask Questions", "Data Preview", "Data Quality"])

    # Tab 1: Auto Dashboard (Enhanced)
    with tab1:
        ui = st.session_state.ui_settings
        st.header(ui["dash_title"] or "Intelligent Dashboard")
        if ui.get("subtitle"):
            st.caption(ui["subtitle"])

        # Schema for all operations
        schema = {
            "columns": df.columns.tolist(),
            "dtypes": {c: str(df[c].dtype) for c in df.columns},
            "nulls": df.isna().sum().to_dict(),
            "shape": df.shape,
        }

        # Generate initial dashboard section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if not st.session_state.get("dashboard_generated", False):
                if st.button("Generate Smart Dashboard", type="primary"):
                    tool_calls = generate_auto_dashboard(df, schema)
                    st.session_state.dashboard_plots = tool_calls
                    st.session_state.dashboard_generated = True
                    st.rerun()
        
        with col2:
            if st.session_state.get("dashboard_generated", False):
                if st.button("Clear Dashboard", type="secondary"):
                    st.session_state.dashboard_plots = []
                    st.session_state.dashboard_generated = False
                    st.rerun()

        # Custom Plot Generation Section
        if st.session_state.get("dashboard_generated", False):
            st.subheader("Add Custom Plot")
            
            with st.expander("Generate custom visualization", expanded=False):
                custom_query = st.text_area(
                    "Describe the plot you want to add:",
                    placeholder="e.g., Show the correlation between price and sales by category",
                    height=80
                )
                
                col_gen, col_help = st.columns([1, 1])
                
                with col_gen:
                    if st.button("Generate & Add Plot", disabled=not custom_query.strip()):
                        with st.spinner("Generating custom plot..."):
                            try:
                                # Use the main analysis chain to generate plot
                                chain = make_chain()
                                resp = chain.invoke({
                                    "question": custom_query, 
                                    "schema": json.dumps(schema)
                                })
                                
                                # Extract tool calls
                                tool_calls = getattr(resp, "tool_calls", None)
                                if not tool_calls and hasattr(resp, "additional_kwargs"):
                                    tool_calls = resp.additional_kwargs.get("tool_calls", [])
                                
                                if tool_calls:
                                    # Add each generated plot to dashboard
                                    new_plots = []
                                    for call in tool_calls:
                                        # Add metadata for custom plots
                                        call["custom"] = True
                                        call["query"] = custom_query
                                        new_plots.append(call)
                                    
                                    st.session_state.dashboard_plots.extend(new_plots)
                                    st.session_state.custom_plots_count += len(new_plots)
                                    st.success(f"Added {len(new_plots)} plot(s) to dashboard!")
                                    st.rerun()
                                else:
                                    st.warning("No visualization was generated from your request. Try rephrasing.")
                                    
                            except Exception as e:
                                st.error(f"Failed to generate custom plot: {str(e)}")
                
                with col_help:
                    st.info("💡 **Example requests:**\n"
                           "- Show sales trends over time\n"
                           "- Compare revenue by region\n"
                           "- Distribution of customer ages\n"
                           "- Correlation between features")

            # Dashboard Management and Reordering
            if st.session_state.dashboard_plots:
                st.subheader("Manage Dashboard")
                
                with st.expander("Customize Dashboard Layout", expanded=True):
                    st.write("**Reorder and manage your dashboard plots:**")
                    
                    # Create a list of plot descriptions for easy identification
                    plot_descriptions = []
                    for i, plot in enumerate(st.session_state.dashboard_plots):
                        plot_name = plot['name']
                        plot_args = plot.get('args', {})
                        
                        # Create readable description
                        if plot.get('custom', False):
                            desc = f"🎨 Custom: {plot.get('query', 'Custom plot')[:50]}..."
                        else:
                            desc = f"📊 {plot_name}"
                            if 'by' in plot_args:
                                desc += f" by {plot_args['by']}"
                            if 'col' in plot_args:
                                desc += f" ({plot_args['col']})"
                            if 'x' in plot_args and 'y' in plot_args:
                                desc += f" ({plot_args['x']} vs {plot_args['y']})"
                        
                        plot_descriptions.append(f"{i+1}. {desc}")
                    
                    # Display current order
                    st.write("**Current order:**")
                    for desc in plot_descriptions:
                        st.write(desc)
                    
                    # Drag and drop reordering
                    # Drag and drop reordering
                    try:
                        from streamlit_sortables import sort_items
                        
                        st.write("**Drag to reorder plots:**")
                        
                        # Create sortable items with readable descriptions
                        sortable_items = [desc.split('. ', 1)[1] for desc in plot_descriptions]
                        
                        sorted_items = sort_items(
                            sortable_items,
                            direction="vertical",
                            key="dashboard_sort"
                        )
                        
                        # Check if order changed and update
                        if sorted_items != sortable_items:
                            # Map sorted items back to original indices
                            original_map = {desc.split('. ', 1)[1]: i for i, desc in enumerate(plot_descriptions)}
                            new_order = [original_map[item] for item in sorted_items]
                            
                            # Reorder the actual plots
                            reordered_plots = [st.session_state.dashboard_plots[i] for i in new_order]
                            st.session_state.dashboard_plots = reordered_plots
                            st.success("Dashboard reordered!")
                            st.rerun()
                    
                    except ImportError:
                        st.info("💡 Install `streamlit-sortables` for drag-and-drop: `pip install streamlit-sortables`")
                        
                        # Fallback to dropdown method
                        col_move, col_remove = st.columns(2)
                        
                        with col_move:
                            st.write("**Move plots:**")
                            if len(st.session_state.dashboard_plots) > 1:
                                move_from = st.selectbox(
                                    "Move plot from position:",
                                    range(1, len(st.session_state.dashboard_plots) + 1),
                                    format_func=lambda x: f"{x}. {plot_descriptions[x-1].split('. ', 1)[1]}"
                                )
                                move_to = st.selectbox(
                                    "To position:",
                                    range(1, len(st.session_state.dashboard_plots) + 1),
                                    index=move_from-1 if move_from <= len(st.session_state.dashboard_plots) else 0
                                )
                                
                                if st.button("Move Plot") and move_from != move_to:
                                    plots = st.session_state.dashboard_plots
                                    plot_to_move = plots.pop(move_from - 1)
                                    plots.insert(move_to - 1, plot_to_move)
                                    st.session_state.dashboard_plots = plots
                                    st.success("Plot moved successfully!")
                                    st.rerun()
                        
                        with col_remove:
                            st.write("**Move plots (fallback):**")
                            # Add fallback remove functionality here if needed
                    
                    # Remove plots section (works for both drag-drop and fallback)
                    st.write("**Remove plots:**")
                    plots_to_remove = st.multiselect(
                        "Select plots to remove:",
                        range(len(st.session_state.dashboard_plots)),
                        format_func=lambda x: plot_descriptions[x],
                        key="remove_plots_multiselect"
                    )
                    
                    if plots_to_remove and st.button("Remove Selected", type="secondary"):
                        # Remove in reverse order to maintain indices
                        for idx in sorted(plots_to_remove, reverse=True):
                            st.session_state.dashboard_plots.pop(idx)
                        st.success(f"Removed {len(plots_to_remove)} plot(s)")
                        st.rerun()

                # Render dashboard in grid
                st.subheader("Your Data Dashboard")
                
                plots_to_show = st.session_state.dashboard_plots
                cols_per_row = int(st.session_state.ui_settings["columns"])

                for i in range(0, len(plots_to_show), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j, col in enumerate(cols):
                        if i + j < len(plots_to_show):
                            plot_data = plots_to_show[i + j]
                            with col:
                                with st.container(border=True):
                                    # Add header for custom plots
                                    if plot_data.get('custom', False):
                                        st.caption(f"🎨 Custom: {plot_data.get('query', '')[:60]}...")
                                    
                                    result = execute_tool_call(df, plot_data)
                                    render_result(result, unique_key=f"dashboard_{i}_{j}")
            else:
                st.info("No visualizations in dashboard. Generate the smart dashboard or add custom plots.")

    # Tab 2: Ask Questions
    with tab2:
        st.header("Ask Questions About Your Data")

        # Question templates
        with st.expander("Example Questions"):
            st.markdown("""
            - What are the top 5 categories by revenue?
            - Show correlation between price and quantity
            - Create a time series of monthly sales
            - Find outliers in the profit column
            - Compare distributions of values across segments
            - What features are most important for predicting churn?
            """)

        question = st.text_input("Ask a question about your data:")

        c1, c2 = st.columns(2)
        with c1:
            run_btn = st.button("Analyze", type="primary", key="analyze_btn")
        with c2:
            st.caption("Model: gemini-2.5-pro (auto-fallback to flash)")

        if run_btn and question:
            model_primary = MODEL_ID
            model_fallback = MODEL_FALLBACK

            def _run(model_id: str):
                chain = make_chain(model_id)
                return chain.invoke({"question": question, "schema": json.dumps(schema)})

            try:
                resp = _run(model_primary)
                used_model = model_primary
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    resp = _run(model_fallback)
                    used_model = model_fallback
                    st.info("Switched to Gemini Flash due to rate limits.")
                else:
                    raise

            # Execute tool calls
            tool_calls = getattr(resp, "tool_calls", None)
            if not tool_calls and hasattr(resp, "additional_kwargs"):
                tool_calls = resp.additional_kwargs.get("tool_calls", [])

            executed_any = False
            for idx, call in enumerate(tool_calls or []):
                executed_any = True
                out = execute_tool_call(df, call)
                render_result(out, unique_key=f"question_{idx}")

            if getattr(resp, "content", None):
                st.caption(f"Model response ({used_model}):")
                st.write(resp.content)

            if not executed_any:
                st.info("No visualization was generated. Try rephrasing your question.")

    # Tab 3: Data Preview
    with tab3:
        st.header("Data Preview & Schema")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("First 10 rows")
            st.dataframe(df.head(10), use_container_width=True)

        with col2:
            st.subheader("Schema Info")
            st.json(schema)

        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)

    # Tab 4: Data Quality Report
    with tab4:
        st.header("Data Quality Analysis")

        if st.button("Generate Quality Report"):
            with st.spinner("Analyzing data quality..."):
                report = data_quality_report(df)
                render_result(report, unique_key="quality_report")

else:
    st.info("Please upload a CSV file to begin analysis")

    # Welcome message
    st.markdown("""
    ### Welcome to Ask-Your-Data

    This AI-powered data analysis tool helps you:
    - Auto-generate dashboards from your data
    - Ask questions in natural language
    - Preprocess data with smart defaults
    - Analyze data quality automatically

    **Get started:** Upload a CSV file in the sidebar.
    """)