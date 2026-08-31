import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# AI MAINTENANCE & OPTIMIZATION
# Step 2: Exploratory Data Analysis
# ============================================================


# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------

DATA_PATH = "data/raw/simulated_iiot_dataset.csv"

df = pd.read_csv(DATA_PATH)


# ------------------------------------------------------------
# 2. Convert timestamp
# ------------------------------------------------------------

df["timestamp"] = pd.to_datetime(df["timestamp"])


# ------------------------------------------------------------
# 3. Basic information
# ------------------------------------------------------------

print("\nDataset shape:")
print(df.shape)

print("\nTimestamp range:")
print(df["timestamp"].min())
print(df["timestamp"].max())


# ------------------------------------------------------------
# 4. Failure distribution
# ------------------------------------------------------------

print("\nMachine failure distribution:")
print(df["machine_failure"].value_counts())


# ------------------------------------------------------------
# 5. Failure percentage
# ------------------------------------------------------------

failure_percentage = df["machine_failure"].mean() * 100

print(f"\nFailure percentage: {failure_percentage:.2f}%")


# ------------------------------------------------------------
# 6. Sensor columns
# ------------------------------------------------------------

sensor_columns = [
    "temperature",
    "vibration",
    "pressure",
    "humidity",
    "rotation_speed",
    "voltage",
    "current",
    "oil_level",
    "load",
    "motor_temperature",
    "gearbox_temperature",
    "sound_level",
    "fan_speed",
    "reactive_power",
    "active_power",
]


# ------------------------------------------------------------
# 7. Plot failure distribution
# ------------------------------------------------------------

plt.figure(figsize=(6, 4))

sns.countplot(
    data=df,
    x="machine_failure"
)

plt.title("Machine Failure Distribution")
plt.xlabel("Machine Failure (0 = Normal, 1 = Failure)")
plt.ylabel("Number of Records")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 8. Sensor distributions
# ------------------------------------------------------------

df[sensor_columns].hist(
    figsize=(16, 12),
    bins=30
)

plt.suptitle("Sensor Value Distributions")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 9. Correlation matrix
# ------------------------------------------------------------

plt.figure(figsize=(14, 10))

correlation_matrix = df[sensor_columns + ["machine_failure"]].corr()

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f"
)

plt.title("Sensor Correlation Matrix")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 10. Sensor comparison with failure
# ------------------------------------------------------------

for column in sensor_columns:

    plt.figure(figsize=(6, 4))

    sns.boxplot(
        data=df,
        x="machine_failure",
        y=column
    )

    plt.title(f"{column} vs Machine Failure")
    plt.xlabel("Machine Failure (0 = Normal, 1 = Failure)")
    plt.ylabel(column)

    plt.tight_layout()
    plt.show()