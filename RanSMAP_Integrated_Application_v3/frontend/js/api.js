
import { CONFIG, STREAMS } from "./config.js";

async function request(path, options = {}) {
    const controller = new AbortController();
    const timeout = setTimeout(
        () => controller.abort(),
        CONFIG.TIMEOUT
    );

    try {
        const response = await fetch(CONFIG.API_BASE_URL + path, {
            ...options,
            signal: controller.signal
        });

        const data = await response.json().catch(() => ({}));

        if (!response.ok) {
            const error = data.detail || data;
            const message =
                error?.message ||
                error?.details?.message ||
                (typeof error === "string" ? error : null) ||
                `HTTP ${response.status}`;

            throw new Error(message);
        }

        return data;
    } catch (error) {
        if (error.name === "AbortError") {
            throw new Error("Request timed out. Please try again.");
        }
        throw error;
    } finally {
        clearTimeout(timeout);
    }
}

function createFormData(files, sampleId, label) {
    const formData = new FormData();

    formData.append("sample_id", sampleId);

    if (label) {
        formData.append("label", label);
    }

    // One ZIP archive OR all six CSV files.
    if (files.archive) {
        formData.append("archive", files.archive);
    } else {
        for (const stream of STREAMS) {
            if (!files[stream]) {
                throw new Error(`Missing CSV file: ${stream}`);
            }

            formData.append(stream, files[stream]);
        }
    }

    return formData;
}

export const api = {
    health() {
        return request("/api/health");
    },

    preprocess(files, sampleId, label) {
        return request("/api/preprocess", {
            method: "POST",
            body: createFormData(files, sampleId, label)
        });
    },

    predict(files, sampleId) {
        return request("/api/predict", {
            method: "POST",
            body: createFormData(files, sampleId)
        });
    },

    adaptivePredict(files, sampleId) {
        return request("/api/adaptive/predict", {
            method: "POST",
            body: createFormData(files, sampleId)
        });
    },

    feedback(predictionId, sampleId, actualLabel) {
        return request("/api/adaptive/feedback", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                prediction_id: predictionId,
                sample_id: sampleId,
                actual_label: actualLabel
            })
        });
    },

    compare(files, sampleId) {
        return request("/api/compare", {
            method: "POST",
            body: createFormData(files, sampleId)
        });
    },

    modelStatus() {
        return request("/api/models/status");
    },

    adaptiveStatus() {
        return request("/api/adaptive/status");
    }
};
