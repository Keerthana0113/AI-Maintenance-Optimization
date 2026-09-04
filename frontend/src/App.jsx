import { useEffect, useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

function App() {
  const [summary, setSummary] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [apiError, setApiError] = useState("");

  const [form, setForm] = useState({
    temperature: 70,
    vibration: 0.5,
    pressure: 7,
    humidity: 50,
    rotation_speed: 1500,
    voltage: 220,
    current: 10,
    oil_level: 70,
    load: 50,
    motor_temperature: 75,
    gearbox_temperature: 65,
    sound_level: 70,
    fan_speed: 1200,
    reactive_power: 2,
    active_power: 10,
  });

  const [prediction, setPrediction] = useState(null);
  const [predicting, setPredicting] = useState(false);

  useEffect(() => {
    loadDashboard();
  }, []);

  async function loadDashboard() {
    try {
      setLoading(true);
      setApiError("");

      const summaryResponse = await fetch(`${API_URL}/data/summary`);

      if (!summaryResponse.ok) {
        throw new Error("Unable to load data summary");
      }

      const summaryData = await summaryResponse.json();
      setSummary(summaryData);

      try {
        const metricsResponse = await fetch(`${API_URL}/model/metrics`);

        if (metricsResponse.ok) {
          const metricsData = await metricsResponse.json();
          setMetrics(metricsData);
        }
      } catch {
        console.log("Metrics endpoint unavailable");
      }
    } catch (error) {
      console.error(error);
      setApiError(
        "Unable to connect to FastAPI. Make sure the backend is running on port 8000."
      );
    } finally {
      setLoading(false);
    }
  }

  function handleChange(event) {
    const { name, value } = event.target;

    setForm({
      ...form,
      [name]: Number(value),
    });
  }

  async function handlePrediction(event) {
    event.preventDefault();

    try {
      setPredicting(true);
      setPrediction(null);

      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(form),
      });

      if (!response.ok) {
        throw new Error("Prediction failed");
      }

      const result = await response.json();

      setPrediction(result);
    } catch (error) {
      console.error(error);

      setPrediction({
        error: "Prediction failed. Check the FastAPI server.",
      });
    } finally {
      setPredicting(false);
    }
  }

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>AI Maintenance Optimization</h1>
          <p>Predictive Maintenance & Machine Failure Monitoring</p>
        </div>

        <div className={`status ${apiError ? "disconnected" : "connected"}`}>
  <span className="status-dot"></span>
  {apiError ? "API Disconnected" : "API Connected"}
</div>
      </header>

      <main className="container">

        {apiError && (
          <div className="error">
            ⚠️ {apiError}
          </div>
        )}

        <section className="stats-grid">

          <div className="card stat-card">
            <span>Total Machines</span>
            <strong>
              {loading ? "..." : summary?.total_records ?? 0}
            </strong>
          </div>

          <div className="card stat-card">
            <span>Normal Records</span>
            <strong className="normal">
              {loading ? "..." : summary?.normal_records ?? 0}
            </strong>
          </div>

          <div className="card stat-card">
            <span>Failure Records</span>
            <strong className="danger">
              {loading ? "..." : summary?.failure_records ?? 0}
            </strong>
          </div>

          <div className="card stat-card">
            <span>Failure Rate</span>
            <strong>
              {loading
                ? "..."
                : `${((summary?.failure_rate ?? 0) * 100).toFixed(1)}%`}
            </strong>
          </div>

        </section>

        <section className="dashboard-grid">

          <div className="card">
            <h2>Machine Failure Prediction</h2>

            <p className="description">
              Enter current machine sensor readings to predict the
              probability of machine failure.
            </p>

            <form onSubmit={handlePrediction}>

              <div className="form-grid">

                {Object.entries(form).map(([key, value]) => (
                  <div className="form-group" key={key}>
                    <label>
                      {key
                        .replaceAll("_", " ")
                        .replace(/\b\w/g, (letter) =>
                          letter.toUpperCase()
                        )}
                    </label>

                    <input
                      type="number"
                      step="any"
                      name={key}
                      value={value}
                      onChange={handleChange}
                    />
                  </div>
                ))}

              </div>

              <button
                type="submit"
                className="predict-button"
                disabled={predicting}
              >
                {predicting
                  ? "Analyzing..."
                  : "Predict Machine Failure"}
              </button>

            </form>
          </div>

          <div className="card prediction-card">
            <h2>Prediction Result</h2>

            {!prediction && (
              <div className="empty">
                <div className="empty-icon">🤖</div>
                <p>
                  Enter sensor values and click
                  <strong> Predict Machine Failure</strong>.
                </p>
              </div>
            )}

            {prediction?.error && (
              <div className="error">
                {prediction.error}
              </div>
            )}

            {prediction && !prediction.error && (
              <>
                <div
                  className={`prediction-status ${
                    prediction.predicted_failure === 1
                      ? "failure"
                      : "safe"
                  }`}
                >
                  {prediction.predicted_failure === 1
                    ? "⚠️ FAILURE RISK"
                    : "✓ MACHINE NORMAL"}
                </div>

                <div className="probability">
                  <span>Failure Probability</span>
                  <strong>
                    {(prediction.failure_probability * 100).toFixed(1)}%
                  </strong>
                </div>

                <div className="risk">
                  Risk Level:
                  <strong>{prediction.risk_level}</strong>
                </div>

                {prediction.recommendation && (
                  <div className="recommendation">
                    <h3>Maintenance Recommendation</h3>
                    <p>{prediction.recommendation}</p>
                  </div>
                )}
              </>
            )}
          </div>

        </section>

        <section className="card">
          <h2>Machine Learning Model Performance</h2>

          {metrics ? (
            <div className="model-grid">

              <div className="model-box">
                <h3>Selected Model</h3>

                <div className="model-name">
                  {metrics.selected_model}
                </div>

                <p>
                  Failure detection model used by the API.
                </p>
              </div>

              {metrics.logistic_regression && (
                <div className="model-box">
                  <h3>Logistic Regression</h3>

                  <p>
                    Accuracy:
                    <strong>
                      {" "}
                      {(metrics.logistic_regression.accuracy * 100).toFixed(1)}%
                    </strong>
                  </p>

                  <p>
                    Recall:
                    <strong>
                      {" "}
                      {(metrics.logistic_regression.recall * 100).toFixed(1)}%
                    </strong>
                  </p>

                  <p>
                    ROC-AUC:
                    <strong>
                      {" "}
                      {metrics.logistic_regression.roc_auc.toFixed(3)}
                    </strong>
                  </p>
                </div>
              )}

              {metrics.random_forest && (
                <div className="model-box">
                  <h3>Random Forest</h3>

                  <p>
                    Accuracy:
                    <strong>
                      {" "}
                      {(metrics.random_forest.accuracy * 100).toFixed(1)}%
                    </strong>
                  </p>

                  <p>
                    Recall:
                    <strong>
                      {" "}
                      {(metrics.random_forest.recall * 100).toFixed(1)}%
                    </strong>
                  </p>

                  <p>
                    ROC-AUC:
                    <strong>
                      {" "}
                      {metrics.random_forest.roc_auc.toFixed(3)}
                    </strong>
                  </p>
                </div>
              )}

            </div>
          ) : (
            <p>Model metrics will appear here.</p>
          )}
        </section>

        <footer>
          <p>
            AI Maintenance Optimization • Predictive Maintenance System
          </p>
        </footer>

      </main>
    </div>
  );
}

export default App;