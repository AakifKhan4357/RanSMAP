
import { api } from "./api.js";
import { STREAMS } from "./config.js";

const $ = (selector) => document.querySelector(selector);

const state = {
    page: "dashboard",
    backendOnline: false,
    history: JSON.parse(sessionStorage.getItem("ransmap_history") || "[]"),
    last: null
};

state.last = state.history.length ? state.history[0] : null;

function escapeHTML(value) {
    return String(value ?? "").replace(/[&<>"']/g, (char) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;"
    })[char]);
}

function showToast(message) {
    const toast = $("#toast");
    if (!toast) return;

    toast.textContent = message;
    toast.classList.add("show");

    setTimeout(() => toast.classList.remove("show"), 3000);
}

function saveHistory() {
    sessionStorage.setItem(
        "ransmap_history",
        JSON.stringify(state.history)
    );
}

function formatPercent(value) {
    const number = Number(value);
    return Number.isFinite(number)
        ? `${(number * 100).toFixed(2)}%`
        : "N/A";
}

function backendStatus() {
    const connection = $("#connection");

    if (!connection) return;

    connection.textContent = state.backendOnline
        ? "● Backend connected"
        : "● Backend disconnected";

    connection.className = state.backendOnline ? "online" : "offline";
}

function renderDashboard() {
    const total = state.history.length;
    const latest = state.last;

    return `
        <div class="grid">
            <div class="card metric">
                <span>Total executions</span>
                <h2>${total}</h2>
            </div>

            <div class="card metric">
                <span>Backend status</span>
                <h2>${state.backendOnline ? "Online" : "Offline"}</h2>
            </div>

            <div class="card metric">
                <span>Latest prediction</span>
                <h2>${latest ? escapeHTML(latest.prediction) : "No data"}</h2>
            </div>
        </div>

        <div class="card">
            <h2>Welcome to RanSMAP</h2>
            <p>
                Ransomware Detection using Storage and Memory
                Behavioral Analysis.
            </p>
            <p>
                Upload the six behavioral CSV files to analyze an
                execution using the integrated baseline model.
            </p>
            <button class="primary" id="open-detection">
                Start detection
            </button>
        </div>

        <div class="card">
            <h3>Latest execution</h3>
            ${
                latest
                    ? `
                        <p><strong>Sample ID:</strong>
                        ${escapeHTML(latest.sample_id)}</p>

                        <p><strong>Prediction:</strong>
                        ${escapeHTML(latest.prediction)}</p>

                        <p><strong>Time:</strong>
                        ${escapeHTML(latest.time)}</p>
                    `
                    : "<p>No predictions have been performed yet.</p>"
            }
        </div>
    `;
}


function renderDetection() {
    const fileInputs = STREAMS.map((stream) => `
        <div class="file-upload">
            <label for="file-${stream}">
                ${escapeHTML(stream)}.csv
            </label>
            <input
                type="file"
                id="file-${stream}"
                accept=".csv"
            >
        </div>
    `).join("");

    return `
        <div class="card">
            <h2>Ransomware Detection</h2>

            <p>
                Upload one ZIP containing the six CSV files, or select
                all six CSV files individually.
            </p>

            <form id="detect-form">
                <div class="field">
                    <label for="sample-id">Sample ID</label>
                    <input
                        type="text"
                        id="sample-id"
                        placeholder="Example: test_sample_001"
                    >
                </div>

                <div class="field">
                    <label for="model-select">Detection Model</label>
                    <select id="model-select">
                        <option value="baseline">
                            Baseline CNN-LSTM
                        </option>
                        <option value="adaptive">
                            Adaptive CNN-LSTM
                        </option>
                    </select>
                </div>

                <h3>Option 1: ZIP archive</h3>

                <div class="file-upload">
                    <label for="file-archive">
                        All six CSVs in one .zip
                    </label>
                    <input
                        type="file"
                        id="file-archive"
                        accept=".zip"
                    >
                </div>

                <h3>Option 2: Behavioral CSV files</h3>

                <div class="grid">
                    ${fileInputs}
                </div>

                <div class="actions">
                    <button
                        type="submit"
                        class="primary"
                        id="run-btn"
                        ${state.backendOnline ? "" : "disabled"}
                    >
                        Run detection
                    </button>

                    <button
                        type="button"
                        class="btn"
                        id="compare-btn"
                        ${state.backendOnline ? "" : "disabled"}
                    >
                        Compare Models
                    </button>
                </div>
            </form>

            <p id="detection-status">
                ${
                    state.backendOnline
                        ? "Backend ready for detection."
                        : "Connect the backend before running detection."
                }
            </p>
        </div>

        <div id="prediction-result"></div>
        <div id="comparison-result"></div>
    `;
}


function renderAdaptive() {
    return `
        <div class="card">
            <h2>Adaptive Learning</h2>

            <p>
                The adaptive detection module provides an inference
                interface for the adaptive CNN-LSTM model.
            </p>

            <div id="adaptive-status" class="status-panel">
                <p>Checking adaptive model status...</p>
            </div>

            <h3>Adaptive model capabilities</h3>
            <ul>
                <li>Adaptive model inference</li>
                <li>Prediction identification</li>
                <li>Feedback submission for labelled executions</li>
                <li>Model status monitoring</li>
            </ul>

            <h3>Current implementation status</h3>
            <p>
                The current integration supports adaptive inference
                and feedback acceptance. Actual incremental model
                updates and replay-based learning depend on the
                adaptive module implementation.
            </p>

            <button id="refresh-adaptive" class="btn btn-primary">
                Refresh Model Status
            </button>
        </div>
    `;
}

async function refreshAdaptiveStatus() {
    const container = $("#adaptive-status");
    const button = $("#refresh-adaptive");

    if (!container) return;

    container.innerHTML = "<p>Checking model status...</p>";

    if (button) button.disabled = true;

    try {
        const result = await api.modelStatus();

        const baseline = result.baseline || {};
        const adaptive = result.adaptive || {};

        container.innerHTML = `
            <h3>Model availability</h3>

            <p>
                <strong>Baseline CNN-LSTM:</strong>
                ${baseline.available ? "Available" : "Unavailable"}
            </p>

            <p>
                <strong>Adaptive CNN-LSTM:</strong>
                ${adaptive.available ? "Available" : "Unavailable"}
            </p>

            <h3>Adaptive model details</h3>

            <p>
                <strong>Readiness:</strong>
                ${adaptive.ready ? "Ready" : "Not ready"}
            </p>

            <p>
                <strong>Model version:</strong>
                ${escapeHTML(String(adaptive.model_version ?? "Not specified"))}
            </p>

            <p>
                <strong>Adaptation status:</strong>
                ${escapeHTML(String(adaptive.adaptation_status ?? "Not specified"))}
            </p>

            <p>
                <strong>Pending feedback:</strong>
                ${Number(adaptive.pending_feedback_count ?? 0)}
            </p>

            <p>
                <strong>Total model updates:</strong>
                ${Number(adaptive.total_updates ?? 0)}
            </p>
        `;
    } catch (error) {
        container.innerHTML = `
            <p class="warn">
                Unable to retrieve model status:
                ${escapeHTML(error.message)}
            </p>
        `;
    } finally {
        if (button) button.disabled = false;
    }
}


function renderFeedback() {
    const eligibleRecords = state.history.filter(
        item => item.prediction_id
    );

    return `
        <div class="card">
            <h2>Detection Feedback</h2>

            <p>
                Provide the actual classification of an adaptive
                model prediction.
            </p>

            ${
                eligibleRecords.length > 0
                    ? `
                        <form id="feedback-form">
                            <div class="field">
                                <label for="feedback-execution">
                                    Select execution
                                </label>

                                <select id="feedback-execution" required>
                                    ${eligibleRecords.map(item => `
                                        <option value="${escapeHTML(item.record_id)}">
                                            ${escapeHTML(item.sample_id)}
                                            — ${escapeHTML(item.prediction)}
                                        </option>
                                    `).join("")}
                                </select>
                            </div>

                            <div class="field">
                                <label for="actual-label">
                                    Actual classification
                                </label>

                                <select id="actual-label" required>
                                    <option value="Benign">Benign</option>
                                    <option value="Ransomware">Ransomware</option>
                                </select>
                            </div>

                            <button type="submit" class="primary">
                                Submit Feedback
                            </button>
                        </form>
                    `
                    : `
                        <p class="warn">
                            No adaptive predictions available.
                            Run an adaptive prediction first.
                        </p>
                    `
            }
        </div>
    `;
}


async function submitFeedback(event) {
    event.preventDefault();

    const recordId = $("#feedback-execution").value;

    const record = state.history.find(
        item => item.record_id === recordId
    );

    if (!record || !record.prediction_id) {
        showToast("No valid adaptive prediction selected.");
        return;
    }

    const actualLabel = $("#actual-label").value;
    const button = $("#feedback-form button[type='submit']");

    button.disabled = true;
    button.textContent = "Submitting...";

    try {
        const result = await api.feedback(
            record.prediction_id,
            record.sample_id,
            actualLabel
        );

        if (!result.feedback_accepted) {
            throw new Error("Backend did not accept the feedback.");
        }

        showToast("Feedback submitted successfully.");

        $("#feedback-form").reset();

    } catch (error) {
        showToast(`Feedback failed: ${error.message}`);
    } finally {
        button.disabled = false;
        button.textContent = "Submit Feedback";
    }
}

function renderHistory() {
    if (!state.history.length) {
        return `
            <div class="card">
                <h2>Execution History</h2>
                <p>No executions recorded in this browser session.</p>
            </div>
        `;
    }

    const rows = state.history.map((item, index) => `
        <tr>
            <td>${index + 1}</td>
            <td>${escapeHTML(item.sample_id)}</td>
            <td>${escapeHTML(item.prediction)}</td>
            <td>${escapeHTML(item.time)}</td>
        </tr>
    `).join("");

    return `
        <div class="card">
            <h2>Execution History</h2>

            <div class="actions">
                <button id="export-history">Export JSON</button>
                <button id="clear-history">Clear history</button>
            </div>

            <div class="table-wrap">
                <table>
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Sample ID</th>
                            <th>Prediction</th>
                            <th>Time</th>
                        </tr>
                    </thead>
                    <tbody>${rows}</tbody>
                </table>
            </div>

            <p>History is stored for this browser session only.</p>
        </div>
    `;
}

function renderResearch() {
    return `
        <div class="card">
            <h2>About RanSMAP</h2>

            <p>
                RanSMAP is a research dataset and platform designed
                for ransomware detection using storage and memory
                access patterns collected during program execution.
            </p>

            <h3>Research motivation</h3>
            <p>
                Ransomware can exhibit suspicious behaviour through
                file-system and storage activity. However, legitimate
                applications may also generate intensive storage
                operations. Memory access patterns provide an additional
                behavioural perspective that can help distinguish
                execution patterns.
            </p>

            <h3>Project objective</h3>
            <p>
                This project explores an adaptive ransomware detection
                framework using a CNN-LSTM model and behavioural features
                extracted from storage and memory activity.
            </p>

            <h3>Project workflow</h3>
            <ol>
                <li>Collect storage and memory behavioural data.</li>
                <li>Upload the six behavioural CSV files.</li>
                <li>Preprocess the execution data.</li>
                <li>Generate fixed-size behavioural feature windows.</li>
                <li>Run the baseline CNN-LSTM prediction model.</li>
                <li>Display the predicted class and probability scores.</li>
                <li>Explore the adaptive detection workflow as it is integrated.</li>
            </ol>

            <h3>Feature configuration</h3>
            <p>
                <strong>300 windows × 23 features</strong> per execution.
            </p>
            <p>
                Each execution is represented as a sequence of
                behavioural feature windows for model inference.
            </p>

            <h3>Behavioural streams</h3>
            <p>
                The input data consists of six behavioural streams:
            </p>
            <ul>
                ${STREAMS.map(stream =>
                    `<li>${escapeHTML(stream)}</li>`
                ).join("")}
            </ul>

            <h3>Detection architecture</h3>
            <p>
                The baseline CNN-LSTM model processes the extracted
                behavioural features to classify an execution as
                Benign or Ransomware.
            </p>

            <h3>Adaptive learning direction</h3>
            <p>
                The proposed extension focuses on incremental learning
                from newly encountered, labelled executions. Its
                objective is to incorporate new behavioural knowledge
                while retaining detection performance on previously
                learned benign and ransomware activities.
            </p>
            <p>
                <strong>Note:</strong> Adaptive learning and model
                updates depend on the adaptive module integration.
                This description represents the project's research
                direction, not a claim that continuous learning is
                currently active.
            </p>
        </div>
    `;
}

function render() {
    const pages = {
        dashboard: ["Dashboard", renderDashboard],
        detection: ["Detection", renderDetection],
        adaptive: ["Adaptive learning", renderAdaptive],
        feedback: ["Feedback", renderFeedback],
        history: ["History", renderHistory],
        research: ["Research / About", renderResearch]
    };

    const [title, pageRenderer] = pages[state.page] || pages.dashboard;

    $("#title").textContent = title;
    $("#view").innerHTML = pageRenderer();

    document.querySelectorAll("#nav button").forEach((button) => {
        button.classList.toggle(
            "active",
            button.dataset.page === state.page
        );
    });

    backendStatus();

    if (state.page === "detection") {
        $("#run-btn").disabled = !state.backendOnline;
        $("#detect-form").addEventListener("submit", runDetection);

        $("#compare-btn")?.addEventListener("click", compareModels);
    }
    
    if (state.page === "dashboard") {
        $("#open-detection").addEventListener("click", () => {
            go("detection");
        });
    }

    if (state.page === "history") {
        $("#export-history")?.addEventListener("click", exportHistory);
        $("#clear-history")?.addEventListener("click", clearHistory);
    }
    if (state.page === "adaptive") {
        $("#refresh-adaptive")?.addEventListener(
            "click",
            refreshAdaptiveStatus
        );

        refreshAdaptiveStatus();
    }
    if (state.page === "feedback") {
        $("#feedback-form")?.addEventListener(
            "submit",
            submitFeedback
        );
    }
}

function go(page) {
    state.page = page;
    render();
}

async function checkBackend() {
    try {
        const result = await api.health();
        state.backendOnline = result.status === "healthy";
    } catch (error) {
        state.backendOnline = false;
    }

    backendStatus();

    if (state.page === "detection") {
        $("#run-btn").disabled = !state.backendOnline;

        $("#detection-status").textContent = state.backendOnline
            ? "Backend ready for detection."
            : "Backend disconnected. Start the FastAPI server.";
    }

    if (state.page === "dashboard") {
        render();
    }
}


function collectUploadFiles() {
    const zipFile = $("#file-archive")?.files?.[0] || null;
    const csvInputs = STREAMS.map((stream) => $(`#file-${stream}`));
    const anyCsv = csvInputs.some((input) => input?.files?.length);

    if (zipFile) {
        if (anyCsv) {
            showToast("Choose either one ZIP or six CSV files, not both.");
            return null;
        }

        if (!zipFile.name.toLowerCase().endsWith(".zip")) {
            showToast("The archive must be a .zip file.");
            return null;
        }

        return { archive: zipFile };
    }

    const files = {};

    for (const stream of STREAMS) {
        const file = $(`#file-${stream}`)?.files?.[0];

        if (!file) {
            showToast(`Please select ${stream}.csv, or upload one ZIP.`);
            return null;
        }

        if (!file.name.toLowerCase().endsWith(".csv")) {
            showToast(`Invalid file for ${stream}.`);
            return null;
        }

        // Warn if the file name does not match the slot (e.g. mem_read in the mem_exec slot).
        if (file.name.toLowerCase() !== `${stream}.csv`) {
            if (!confirm(`"${file.name}" is in the ${stream}.csv slot. Continue?`)) {
                return null;
            }
        }

        files[stream] = file;
    }

    return files;
}


function renderWarnings(warnings) {
    if (!warnings?.length) {
        return "";
    }

    return `
        <div class="card warn">
            <h3>Data quality warning</h3>
            ${warnings.map((w) => `<p>${escapeHTML(w)}</p>`).join("")}
        </div>
    `;
}


async function runDetection(event) {
    event.preventDefault();

    if (!state.backendOnline) {
        showToast("Backend is not connected.");
        return;
    }

    const files = collectUploadFiles();

    if (!files) {
        return;
    }

    const sampleId = $("#sample-id").value.trim()
        || `execution_${Date.now()}`;

    const selectedModel = $("#model-select").value;

    const button = $("#run-btn");
    const status = $("#detection-status");
    const resultBox = $("#prediction-result");

    button.disabled = true;
    button.textContent = "Analyzing...";
    status.textContent = "Uploading files and running prediction...";
    resultBox.innerHTML = "";

    try {
        const result = selectedModel === "adaptive"
            ? await api.adaptivePredict(files, sampleId)
            : await api.predict(files, sampleId);

        if (result.status !== "success") {
            throw new Error(
                "The backend did not return a successful prediction."
            );
        }

        const record = {
            record_id: `local-${Date.now()}`,
            sample_id: result.sample_id || sampleId,
            prediction_id: result.prediction_id || null,
            prediction: result.prediction,
            benign_probability: result.benign_probability,
            ransomware_probability: result.ransomware_probability,
            num_windows: result.num_windows,
            num_features: result.num_features,
            threshold: result.threshold,
            model: selectedModel === "adaptive"
                ? "Adaptive CNN-LSTM"
                : "Baseline CNN-LSTM",
            time: new Date().toLocaleString()
        };

        state.history.unshift(record);
        state.last = record;

        saveHistory();

        displayPrediction(record);
        resultBox.insertAdjacentHTML("beforeend", renderWarnings(result.warnings));

        status.textContent = "Prediction completed successfully.";
        showToast("Detection completed.");

    } catch (error) {
        status.textContent = "Detection failed.";

        resultBox.innerHTML = `
            <div class="card warn">
                <h3>Prediction error</h3>
                <p>${escapeHTML(error.message)}</p>
            </div>
        `;

        showToast(error.message);

    } finally {
        button.disabled = !state.backendOnline;
        button.textContent = "Run detection";
    }
}


async function compareModels() {
    if (!state.backendOnline) {
        showToast("Backend is not connected.");
        return;
    }

    const files = collectUploadFiles();

    if (!files) {
        return;
    }

    const sampleId = $("#sample-id").value.trim()
        || `comparison_${Date.now()}`;

    const button = $("#compare-btn");
    const resultBox = $("#comparison-result");

    button.disabled = true;
    button.textContent = "Comparing...";
    resultBox.innerHTML = `
        <div class="card">
            <p>Running baseline and adaptive predictions...</p>
        </div>
    `;

    try {
        const result = await api.compare(files, sampleId);

        if (result.status !== "success") {
            throw new Error("Model comparison was unsuccessful.");
        }

        const adaptive = result.adaptive;
        const baseline = result.baseline;

        // Save the adaptive prediction so feedback can be submitted later.
        const record = {
            record_id: `local-${Date.now()}`,
            sample_id: result.sample_id || sampleId,
            prediction_id: adaptive.prediction_id || null,
            prediction: adaptive.prediction,
            benign_probability: adaptive.benign_probability,
            ransomware_probability: adaptive.ransomware_probability,
            num_windows: result.preprocessing.num_windows,
            num_features: result.preprocessing.num_features,
            threshold: adaptive.threshold,
            model: "Adaptive CNN-LSTM (Comparison)",
            time: new Date().toLocaleString()
        };

        if (record.prediction_id) {
            state.history.unshift(record);
            state.last = record;
            saveHistory();
        }

        renderComparison(result);
        resultBox.insertAdjacentHTML("beforeend", renderWarnings(result.warnings));

        showToast("Model comparison completed.");

    } catch (error) {
        resultBox.innerHTML = `
            <div class="card warn">
                <h3>Comparison failed</h3>
                <p>${escapeHTML(error.message)}</p>
            </div>
        `;

        showToast(`Comparison failed: ${error.message}`);

    } finally {
        button.disabled = !state.backendOnline;
        button.textContent = "Compare Models";
    }
}


function renderComparison(result) {
    const baseline = result.baseline;
    const adaptive = result.adaptive;
    const comparison = result.comparison;
    const preprocessing = result.preprocessing;

    const predictionCard = (model, data) => `
        <div class="card">
            <h3>${escapeHTML(model)}</h3>

            <h2 class="${
                data.prediction === "Ransomware" ? "danger" : "success"
            }">
                ${escapeHTML(data.prediction)}
            </h2>

            <p>
                <strong>Benign probability:</strong>
                ${formatPercent(data.benign_probability)}
            </p>

            <p>
                <strong>Ransomware probability:</strong>
                ${formatPercent(data.ransomware_probability)}
            </p>

            <p>
                <strong>Confidence:</strong>
                ${formatPercent(data.confidence)}
            </p>

            <p>
                <strong>Threshold:</strong>
                ${escapeHTML(data.threshold)}
            </p>
        </div>
    `;

    $("#comparison-result").innerHTML = `
        <div class="card">
            <h2>Model Comparison Results</h2>

            <p>
                <strong>Sample ID:</strong>
                ${escapeHTML(result.sample_id)}
            </p>

            <div class="grid">
                ${predictionCard("Baseline CNN-LSTM", baseline)}
                ${predictionCard("Adaptive CNN-LSTM", adaptive)}
            </div>

            <h3>Comparison Summary</h3>

            <p>
                <strong>Prediction agreement:</strong>
                ${comparison.predictions_match
                    ? "Both models agree"
                    : "Models disagree"}
            </p>

            <p>
                <strong>Same checkpoint initialization:</strong>
                ${comparison.same_checkpoint_initialization
                    ? "Yes"
                    : "No"}
            </p>

            <p>
                <strong>Incremental model updates enabled:</strong>
                ${comparison.model_updates_enabled ? "Yes" : "No"}
            </p>

            <h3>Preprocessing Summary</h3>

            <p>
                <strong>Windows:</strong>
                ${escapeHTML(preprocessing.num_windows)}
            </p>

            <p>
                <strong>Features:</strong>
                ${escapeHTML(preprocessing.num_features)}
            </p>

            <p>
                <strong>Adaptive prediction ID:</strong>
                ${escapeHTML(adaptive.prediction_id || "Not available")}
            </p>

            <p>
                <small>
                    This comparison evaluates the two models on the
                    same execution. It does not demonstrate incremental
                    learning or model improvement.
                </small>
            </p>
        </div>
    `;
}


function displayPrediction(record) {
    const isRansomware = record.prediction === "Ransomware";

    return $("#prediction-result").innerHTML = `
        <div class="card">
            <h2>Prediction Result</h2>

            <h1 class="${isRansomware ? "danger" : "success"}">
                ${escapeHTML(record.prediction)}
            </h1>

            <p>
                Sample ID:
                <strong>${escapeHTML(record.sample_id)}</strong>
            </p>

            <p>
                Detection Model:
                <strong>${escapeHTML(record.model || "Baseline CNN-LSTM")}</strong>
            </p>

            <div class="grid">
                <div class="card metric">
                    <span>Benign probability</span>
                    <h2>${formatPercent(record.benign_probability)}</h2>
                </div>

                <div class="card metric">
                    <span>Ransomware probability</span>
                    <h2>${formatPercent(record.ransomware_probability)}</h2>
                </div>

                <div class="card metric">
                    <span>Number of windows</span>
                    <h2>${escapeHTML(record.num_windows)}</h2>
                </div>

                <div class="card metric">
                    <span>Number of features</span>
                    <h2>${escapeHTML(record.num_features)}</h2>
                </div>
            </div>

            <p>
                Decision threshold:
                ${escapeHTML(record.threshold)}
            </p>

            <p>
                <small>
                    Prediction generated by the integrated
                    ${escapeHTML(record.model || "Baseline CNN-LSTM")} model.
                </small>
            </p>
        </div>
    `;
}


function exportHistory() {
    const blob = new Blob(
        [JSON.stringify(state.history, null, 2)],
        { type: "application/json" }
    );

    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");

    link.href = url;
    link.download = "ransmap_execution_history.json";
    link.click();

    URL.revokeObjectURL(url);
    showToast("History exported.");
}

function clearHistory() {
    if (!confirm("Clear all execution history for this session?")) {
        return;
    }

    state.history = [];
    state.last = null;

    saveHistory();
    render();

    showToast("History cleared.");
}

function setupTheme() {
    const savedTheme = localStorage.getItem("ransmap_theme") || "light";

    document.documentElement.dataset.theme = savedTheme;

    const themeButton = $("#theme");

    function updateThemeButton() {
        themeButton.textContent =
            document.documentElement.dataset.theme === "dark"
                ? "Light mode"
                : "Dark mode";
    }

    updateThemeButton();

    themeButton.addEventListener("click", () => {
        const current = document.documentElement.dataset.theme;
        const next = current === "dark" ? "light" : "dark";

        document.documentElement.dataset.theme = next;
        localStorage.setItem("ransmap_theme", next);

        updateThemeButton();
    });
}

function setupNavigation() {
    $("#nav").addEventListener("click", (event) => {
        const button = event.target.closest("button[data-page]");

        if (!button) return;

        go(button.dataset.page);
    });
}

async function init() {
    setupTheme();
    setupNavigation();

    render();

    await checkBackend();
}

init();
