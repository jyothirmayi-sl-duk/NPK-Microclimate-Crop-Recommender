
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Project directories
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "raw" / "Crop_recommendation.csv"

OUTPUT_DIR = BASE_DIR / "reports" / "visualizations"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Visualization style
sns.set_theme(style="whitegrid")


def load_data():
    """Load crop recommendation dataset."""

    data = pd.read_csv(DATA_PATH)

    print("Dataset loaded successfully!")
    print("Dataset shape:", data.shape)
python -m src.visualizer
    return data



def crop_distribution(data):
    """Visualize the number of records for each crop."""

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
    plt.savefig(OUTPUT_DIR / "crop_distribution.png")
    plt.show()
    plt.close()


def nutrient_analysis(data):
    """Visualize NPK nutrient distribution."""

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
    plt.savefig(OUTPUT_DIR / "npk_distribution.png")
    plt.show()
    plt.close()


def climate_analysis(data):
    """Visualize climate conditions."""

    climate_columns = [
        "temperature",
        "humidity",
        "rainfall"
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

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

    plt.savefig(OUTPUT_DIR / "climate_distribution.png")
    plt.show()
    plt.close()


def correlation_heatmap(data):
    """Display correlation between numerical features."""

    numeric_data = data.select_dtypes(include="number")

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
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png")
    plt.show()
    plt.close()


def feature_relationships(data):
    """Visualize relationships between important features."""

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
    plt.savefig(OUTPUT_DIR / "temperature_humidity.png")
    plt.show()
    plt.close()


def generate_visualizations():
    """Run all visualization functions."""

    data = load_data()

    crop_distribution(data)

    nutrient_analysis(data)

    climate_analysis(data)

    correlation_heatmap(data)

    feature_relationships(data)

    print("\nAll visualizations generated successfully!")
    print("Saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    generate_visualizations()
