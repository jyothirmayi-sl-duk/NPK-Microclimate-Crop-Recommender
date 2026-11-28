
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ============================================================
# 1. PROJECT PATHS AND SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "raw" / "Crop_recommendation.csv"

OUTPUT_DIR = BASE_DIR / "reports" / "visualizations"
FIGURES_DIR = BASE_DIR / "reports" / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

REQUIRED_COLUMNS = [
    "N", "P", "K",
    "temperature", "humidity",
    "ph", "rainfall", "label"
]


# ============================================================
# 2. DATA VALIDATION AND LOADING
# ============================================================

def validate_data(data):
    """Check whether the input DataFrame has required columns."""

    missing = [
        column for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if data.empty:
        raise ValueError("The input dataset is empty.")

    return data.copy()


def load_data():

    data = pd.read_csv(DATA_PATH)
    data = validate_data(data)

    print("Dataset loaded successfully!")
    print("Dataset shape:", data.shape)

    return data


# ============================================================
# 3. EXISTING VISUALIZATION 1 - CROP DISTRIBUTION
# ============================================================

def crop_distribution(data):

    plt.figure(figsize=(12, 6))

    sns.countplot(
        data=data,
        x="label",
        order=data["label"].value_counts().index
    )

    plt.title("Crop Distribution")
    plt.xlabel("Crop Type")
    plt.ylabel("Number of Records")
    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "crop_distribution.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 4. EXISTING VISUALIZATION 2 - NPK DISTRIBUTION
# ============================================================

def nutrient_analysis(data):

    nutrient_columns = ["N", "P", "K"]

    plt.figure(figsize=(12, 6))

    for column in nutrient_columns:

        sns.histplot(
            data=data,
            x=column,
            kde=True,
            label=column,
            alpha=0.4
        )

    plt.title("NPK Nutrient Distribution")
    plt.xlabel("Nutrient Values")
    plt.ylabel("Frequency")
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "npk_distribution.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 5. EXISTING VISUALIZATION 3 - CLIMATE DISTRIBUTION
# ============================================================

def climate_analysis(data):

    climate_columns = [
        "temperature",
        "humidity",
        "rainfall"
    ]

    fig, axes = plt.subplots(
        1, 3,
        figsize=(15, 5)
    )

    for column, axis in zip(climate_columns, axes):

        sns.histplot(
            data=data,
            x=column,
            kde=True,
            ax=axis
        )

        axis.set_title(column.capitalize())

    plt.suptitle("Climate Feature Distribution")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "climate_distribution.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 6. EXISTING VISUALIZATION 4 - CORRELATION HEATMAP
# ============================================================

def correlation_heatmap(data):

    numeric_data = data.select_dtypes(
        include="number"
    )

    plt.figure(figsize=(10, 7))

    sns.heatmap(
        numeric_data.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5
    )

    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "correlation_heatmap.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 7. EXISTING VISUALIZATION 5 - FEATURE RELATIONSHIP
# ============================================================

def feature_relationships(data):

    plt.figure(figsize=(9, 6))

    sns.scatterplot(
        data=data,
        x="temperature",
        y="humidity",
        hue="label",
        alpha=0.7,
        legend=False
    )

    plt.title("Temperature vs Humidity")
    plt.xlabel("Temperature")
    plt.ylabel("Humidity")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "temperature_humidity.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 8. EXISTING VISUALIZATION 6 - GROUPED NPK BAR CHART
# ============================================================

def grouped_npk_bar_chart(data):

    selected_crops = [
        "rice",
        "coconut",
        "coffee",
        "banana",
        "blackgram"
    ]

    crop_data = data[
        data["label"].str.lower().isin(selected_crops)
    ]

    if crop_data.empty:
        raise ValueError(
            "None of the selected crops were found in the dataset."
        )

    mean_npk = (
        crop_data
        .groupby("label")[["N", "P", "K"]]
        .mean()
        .reindex(selected_crops)
        .dropna(how="all")
    )

    ax = mean_npk.plot(
        kind="bar",
        figsize=(11, 6),
        width=0.8
    )

    ax.set_title(
        "Mean NPK Requirements Across Selected Crops"
    )

    ax.set_xlabel("Crop Type")
    ax.set_ylabel("Mean Nutrient Value")

    plt.xticks(rotation=30)
    plt.legend(title="Nutrients")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "grouped_npk_bar.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 9. EXISTING VISUALIZATION 7 - NPK RAINFALL PAIRPLOT
# ============================================================

def npk_rainfall_pairplot(data):

    selected_crops = [
        "rice",
        "coconut",
        "coffee",
        "banana",
        "blackgram"
    ]

    pair_data = data[
        data["label"].str.lower().isin(selected_crops)
    ]

    if pair_data.empty:
        raise ValueError(
            "No selected crop records available for pairplot."
        )

    pairplot = sns.pairplot(
        data=pair_data,
        vars=["N", "P", "K", "rainfall"],
        hue="label",
        diag_kind="hist",
        corner=True,
        plot_kws={"alpha": 0.6, "s": 25}
    )

    pairplot.fig.suptitle(
        "NPK and Rainfall Relationships by Crop",
        y=1.02
    )

    pairplot.savefig(
        OUTPUT_DIR / "npk_rainfall_pairplot.png",
        dpi=300
    )

    plt.close("all")


# ============================================================
# 10. EXISTING VISUALIZATION 8 - INTERACTIVE 3D DASHBOARD
# ============================================================

def create_3d_dashboard(data):

    fig = px.scatter_3d(
        data,
        x="N",
        y="P",
        z="rainfall",
        color="label",
        hover_data=[
            "K",
            "temperature",
            "humidity",
            "ph"
        ],
        title="Interactive 3D Crop Suitability Dashboard",
        labels={
            "N": "Nitrogen",
            "P": "Phosphorus",
            "rainfall": "Rainfall"
        }
    )

    initial_n = float(data["N"].median())
    initial_p = float(data["P"].median())
    initial_rainfall = float(data["rainfall"].median())

    fig.add_trace(
        go.Scatter3d(
            x=[initial_n],
            y=[initial_p],
            z=[initial_rainfall],
            mode="markers",
            marker=dict(
                size=9,
                color="black",
                symbol="diamond"
            ),
            name="Test Soil Input",
            hovertemplate=(
                "Test Soil Input<br>"
                "N: %{x:.2f}<br>"
                "P: %{y:.2f}<br>"
                "Rainfall: %{z:.2f}"
                "<extra></extra>"
            )
        )
    )

    fig.update_layout(
        scene=dict(
            xaxis_title="Nitrogen (N)",
            yaxis_title="Phosphorus (P)",
            zaxis_title="Rainfall"
        ),
        height=750,
        legend_title="Crop Type"
    )

    html_content = fig.to_html(
        full_html=True,
        include_plotlyjs=True,
        div_id="crop_dashboard"
    )

    test_controls = f"""
    <div style="font-family:Arial; padding:20px;
                max-width:850px; margin:auto;">

        <h2>Test Soil Input Controls</h2>

        <label>Nitrogen (N):
            <span id="n_value">{initial_n:.2f}</span>
        </label>

        <input id="n_slider" type="range"
            min="{data['N'].min()}"
            max="{data['N'].max()}"
            step="0.1"
            value="{initial_n}"
            style="width:100%;">

        <br><br>

        <label>Phosphorus (P):
            <span id="p_value">{initial_p:.2f}</span>
        </label>

        <input id="p_slider" type="range"
            min="{data['P'].min()}"
            max="{data['P'].max()}"
            step="0.1"
            value="{initial_p}"
            style="width:100%;">

        <br><br>

        <label>Rainfall:
            <span id="r_value">{initial_rainfall:.2f}</span>
        </label>

        <input id="r_slider" type="range"
            min="{data['rainfall'].min()}"
            max="{data['rainfall'].max()}"
            step="0.1"
            value="{initial_rainfall}"
            style="width:100%;">

        <p>
            Move the sliders to change the test soil point
            in the 3D visualization.
        </p>

    </div>

    <script>

    const nSlider = document.getElementById("n_slider");
    const pSlider = document.getElementById("p_slider");
    const rSlider = document.getElementById("r_slider");

    const testTraceIndex = {len(fig.data) - 1};

    function updateTestPoint() {{

        const n = Number(nSlider.value);
        const p = Number(pSlider.value);
        const r = Number(rSlider.value);

        document.getElementById("n_value").textContent =
            n.toFixed(2);

        document.getElementById("p_value").textContent =
            p.toFixed(2);

        document.getElementById("r_value").textContent =
            r.toFixed(2);

        Plotly.restyle(
            "crop_dashboard",
            {{x: [[n]], y: [[p]], z: [[r]]}},
            [testTraceIndex]
        );
    }}

    nSlider.addEventListener("input", updateTestPoint);
    pSlider.addEventListener("input", updateTestPoint);
    rSlider.addEventListener("input", updateTestPoint);

    </script>
    """

    html_content = html_content.replace(
        "</body>",
        test_controls + "</body>"
    )

    dashboard_path = OUTPUT_DIR / "interactive_dashboard.html"

    with open(dashboard_path, "w", encoding="utf-8") as file:
        file.write(html_content)

    print("3D dashboard saved:", dashboard_path)


# ============================================================
# 11. Q10 - MATPLOTLIB DUAL-AXIS AND SHADED ENVELOPE
# ============================================================

def q10_dual_axis_analysis(data):

    crop_stats = (
        data.groupby("label")
        .agg(
            mean_rainfall=("rainfall", "mean"),
            std_rainfall=("rainfall", "std"),
            sample_count=("rainfall", "count"),
            mean_temperature=("temperature", "mean")
        )
        .reset_index()
    )

    crop_stats["std_rainfall"] = (
        crop_stats["std_rainfall"].fillna(0)
    )

    crop_stats["sem"] = (
        crop_stats["std_rainfall"] /
        crop_stats["sample_count"].pow(0.5)
    )

    crop_stats["lower_ci"] = (
        crop_stats["mean_rainfall"] -
        1.96 * crop_stats["sem"]
    )

    crop_stats["upper_ci"] = (
        crop_stats["mean_rainfall"] +
        1.96 * crop_stats["sem"]
    )

    crop_stats = crop_stats.sort_values(
        "mean_rainfall"
    ).reset_index(drop=True)

    x = range(len(crop_stats))

    fig, ax1 = plt.subplots(figsize=(15, 7))

    ax2 = ax1.twinx()

    # Primary axis: mean rainfall
    ax1.plot(
        x,
        crop_stats["mean_rainfall"],
        marker="o",
        linewidth=2,
        label="Mean Rainfall"
    )

    # Shaded confidence interval
    ax1.fill_between(
        x,
        crop_stats["lower_ci"],
        crop_stats["upper_ci"],
        alpha=0.2,
        label="Approx. 95% Confidence Interval"
    )

    # Secondary axis: mean temperature
    ax2.plot(
        x,
        crop_stats["mean_temperature"],
        color="red",
        linestyle="--",
        marker="s",
        label="Mean Temperature"
    )

    # Annotate highest rainfall crop
    peak_index = crop_stats["mean_rainfall"].idxmax()

    peak_rainfall = crop_stats.loc[
        peak_index, "mean_rainfall"
    ]

    peak_crop = crop_stats.loc[
        peak_index, "label"
    ]

    ax1.annotate(
        f"Highest rainfall: {peak_crop}",
        xy=(peak_index, peak_rainfall),
        xytext=(10, 25),
        textcoords="offset points",
        arrowprops=dict(
            arrowstyle="->",
            color="black"
        ),
        fontsize=9
    )

    ax1.set_xticks(list(x))
    ax1.set_xticklabels(
        crop_stats["label"],
        rotation=60,
        ha="right"
    )

    ax1.set_xlabel("Crop Type")
    ax1.set_ylabel("Mean Rainfall")
    ax2.set_ylabel("Mean Temperature")

    ax1.set_title(
        "Crop-wise Rainfall and Temperature Comparison",
        fontsize=15,
        fontweight="bold"
    )

    ax1.grid(True, alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper left"
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / "Q10_dual_axis_analysis.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Q10 figure saved successfully.")


# ============================================================
# 12. Q11 - SEABORN MULTI-PANEL STATISTICAL DISTRIBUTIONS
# ============================================================

def q11_statistical_distributions(data):

    numeric_columns = [
        "N", "P", "K",
        "temperature", "humidity",
        "ph", "rainfall"
    ]

    fig, axes = plt.subplots(
        1, 2,
        figsize=(19, 8)
    )

    # Panel 1: Correlation heatmap
    correlation = data[numeric_columns].corr()

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        center=0,
        linewidths=0.5,
        ax=axes[0]
    )

    axes[0].set_title(
        "Correlation Heatmap",
        fontweight="bold"
    )

    # Panel 2: Crop-wise rainfall boxplot
    sns.boxplot(
        data=data,
        x="label",
        y="rainfall",
        palette="Set3",
        ax=axes[1]
    )

    axes[1].set_title(
        "Rainfall Distribution Across Crops",
        fontweight="bold"
    )

    axes[1].set_xlabel("Crop Type")
    axes[1].set_ylabel("Rainfall")

    axes[1].tick_params(
        axis="x",
        rotation=60
    )

    fig.suptitle(
        "Statistical Analysis of Crop Recommendation Dataset",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout()

    fig.savefig(
        FIGURES_DIR / "Q11_statistical_distributions.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Q11 figure saved successfully.")


# ============================================================
# 13. Q12 - STANDALONE INTERACTIVE PLOTLY DASHBOARD
# ============================================================

def q12_interactive_dashboard(data):

    fig = px.scatter(
        data,
        x="rainfall",
        y="N",
        color="label",
        hover_data=[
            "P",
            "K",
            "temperature",
            "humidity",
            "ph"
        ],
        title="Interactive Crop Suitability Dashboard",
        labels={
            "rainfall": "Rainfall",
            "N": "Nitrogen"
        }
    )

    # Add range slider
    fig.update_xaxes(
        rangeslider_visible=True,
        title="Rainfall"
    )

    fig.update_yaxes(
        title="Nitrogen (N)"
    )

    fig.update_layout(
        height=750,
        template="plotly_white",
        legend_title="Crop Type",
        hovermode="closest"
    )
    crop_names = sorted(data["label"].unique())
    buttons = [
        dict(
            label="All Crops",
            method="update",
            args=[
                {"visible": [True] * len(crop_names)},
                {"title": "All Crops - Interactive Dashboard"}
            ]
        )
    ]
    fig = go.Figure()
    for crop in crop_names:
        crop_data = data[
            data["label"] == crop
        ]
        fig.add_trace(
            go.Scatter(
                x=crop_data["rainfall"],
                y=crop_data["N"],
                mode="markers",
                name=crop,
                customdata=crop_data[
                    ["P", "K", "temperature", "humidity", "ph"]
                ].values,
                hovertemplate=(
                    "Crop: " + crop +
                    "<br>Rainfall: %{x:.2f}" +
                    "<br>Nitrogen: %{y:.2f}" +
                    "<br>Phosphorus: %{customdata[0]:.2f}" +
                    "<br>Potassium: %{customdata[1]:.2f}" +
                    "<br>Temperature: %{customdata[2]:.2f}" +
                    "<br>Humidity: %{customdata[3]:.2f}" +
                    "<br>pH: %{customdata[4]:.2f}" +
                    "<extra></extra>"
                )
            )
        )
    buttons = [     #filtering part
        dict(
            label="All Crops",
            method="update",
            args=[
                {"visible": [True] * len(crop_names)},
                {"title": "All Crops - Interactive Dashboard"}
            ]
        )
    ]
    for index, crop in enumerate(crop_names):
        visibility = [
            i == index for i in range(len(crop_names))
        ]
        buttons.append(
            dict(
                label=crop,
                method="update",
                args=[
                    {"visible": visibility},
                    {"title": f"Crop Analysis: {crop}"}
                ]
            )
        )
    fig.update_layout(
        title="Interactive Crop Suitability Dashboard",
        xaxis=dict(
            title="Rainfall",
            rangeslider=dict(visible=True)
        ),
        yaxis_title="Nitrogen (N)",
        height=750,
        template="plotly_white",
        hovermode="closest",
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                x=1.02,
                y=1.15,
                xanchor="left",
                yanchor="top"
            )
        ],
        margin=dict(
            l=60,
            r=180,
            t=100,
            b=60
        )
    )
    dashboard_path = (
        BASE_DIR / "reports" / "interactive_dashboard.html"
    )
    dashboard_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )
    fig.write_html(
        dashboard_path,
        full_html=True,
        include_plotlyjs=True,
        auto_open=False
    )

    print("Q12 dashboard saved:", dashboard_path)
def generate_visualizations(data=None):  #visualization part
    if data is None:
        data = load_data()
    data = validate_data(data)
    print("\nGenerating existing visualizations...")
    crop_distribution(data)
    nutrient_analysis(data)
    climate_analysis(data)
    correlation_heatmap(data)
    feature_relationships(data)
    grouped_npk_bar_chart(data)
    npk_rainfall_pairplot(data)
    create_3d_dashboard(data)

    print("\nGenerating Phase 6 visualizations...")
    q10_dual_axis_analysis(data)
    q11_statistical_distributions(data)
    q12_interactive_dashboard(data)
    print("\nAll visualizations generated successfully!")
    print("Existing output directory:", OUTPUT_DIR)
    print("Phase 6 figures directory:", FIGURES_DIR)
if __name__ == "__main__":    #excution
    generate_visualizations()
