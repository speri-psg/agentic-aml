"""Segmentation Agent — K-Means cluster analysis and dynamic segmentation tree."""

from .base_agent import BaseAgent

# OpenAI function-calling format (matches the fine-tuning training data)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "cluster_analysis",
            "description": (
                "Perform dynamic segmentation using K-Means cluster analysis on customer data. "
                "Uses numeric features (avg transactions, amounts, income, balance, age) and "
                "categorical features (account type, gender, age category, channel, NNM, OFAC, 314b). "
                "Alert labels (FP, FN, ALERT) are excluded so clusters reflect natural behavior profiles. "
                "Use n_clusters=0 to auto-select the optimal K via the elbow method."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_type": {
                        "type": "string",
                        "enum": ["Business", "Individual", "All"],
                        "description": "Which customer segment to cluster.",
                    },
                    "n_clusters": {
                        "type": "integer",
                        "description": (
                            "Number of K-Means clusters (2–8), or 0 to auto-select via elbow method. "
                            "Default is 4. Set to exactly the number the user requests "
                            "(e.g. 'cluster into 3' → n_clusters=3, 'show 2 clusters' → n_clusters=2)."
                        ),
                    },
                },
                "required": ["customer_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "alerts_distribution",
            "description": "Show total alerts and false positives distribution across segments.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "prepare_segmentation_data",
            "description": (
                "Process raw customer, account, relationship, and transaction files from ss_files/ "
                "and produce a flat CSV at docs/ds_segmentation_data.csv ready for clustering. "
                "Computes transaction aggregates: avg_trxns_week, avg_trxn_amt, avg_monthly_trxn_amt, "
                "trxn_count, total_trxn_amt, max_trxn_amt, std_trxn_amt. "
                "Call this before running ds_cluster_analysis on new raw data."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cluster_pca_plot",
            "description": (
                "Render a PCA scatter plot visualizing the most recent K-Means clustering "
                "in 2D principal component space. Use ONLY when the user explicitly asks for "
                "the PCA, scatter plot, scatter view, or 2D projection of the clusters. "
                "Most AML analysts do not need this; it is bandwidth-heavy at scale (100K+ "
                "customers). Do NOT call it for general clustering questions — use "
                "ds_cluster_analysis for those."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_type": {
                        "type": "string",
                        "enum": ["Business", "Individual", "All"],
                        "description": "Which customer segment to render. Defaults to the most recent clustering segment if omitted.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ds_cluster_analysis",
            "description": (
                "Perform dynamic segmentation clustering on the ss_files raw data "
                "(customers, accounts, relationships, transactions). "
                "Auto-prepares and joins source data if not already done. "
                "Uses customer demographics (age, gender, citizenship), account features "
                "(account type, balance, account age), and transaction aggregates "
                "(avg transactions/week, avg amount, monthly amount) for K-Means clustering. "
                "Returns a dynamic segmentation treemap and per-cluster statistics. "
                "Use n_clusters=0 to auto-select optimal K via elbow method. "
                "Do NOT call this when the user asks for a PCA scatter — use cluster_pca_plot instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_type": {
                        "type": "string",
                        "enum": ["Business", "Individual", "All"],
                        "description": "Which customer segment to cluster.",
                    },
                    "n_clusters": {
                        "type": "integer",
                        "description": (
                            "Number of K-Means clusters (2–8), or 0 to auto-select via elbow method. "
                            "Default is 4. Set to exactly the number the user requests "
                            "(e.g. 'cluster into 3' → n_clusters=3, 'show 2 clusters' → n_clusters=2). "
                            "IMPORTANT: if the user specifies a number, you MUST pass it here. "
                            "Do NOT use the default 4 when the user has asked for a different count."
                        ),
                    },
                },
                "required": ["customer_type"],
            },
        },
    },
]

SYSTEM_PROMPT = (
    "Look at the given data and respond strictly to the data and no more."
)


class SegmentationAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="segmentation",
            system_prompt=SYSTEM_PROMPT,
            tools=TOOLS,
        )

    def _stream_llm(self, **kwargs):
        # Keep vLLM thinking ON for segmentation — needed for correct oldest/youngest
        # cluster comparisons (numerical reasoning over decimal account ages).
        kwargs.setdefault("extra_body", {})["chat_template_kwargs"] = {"enable_thinking": True}
        return super()._stream_llm(**kwargs)
