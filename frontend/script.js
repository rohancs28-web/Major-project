// Base URL for your FastAPI backend server
const API_BASE_URL = "http://localhost:8000";

/**
 * Triggered when user selects a region/crop and clicks Run Inference
 */
async function fetchPrediction() {
    const region = document.getElementById('regionSelect').value;
    const crop = document.getElementById('cropSelect').value;
    
    const yieldOutput = document.getElementById('yieldOutput');
    yieldOutput.innerHTML = `<i class="fa-solid fa-spinner fa-spin text-sm"></i> Computing...`;

    try {
        // Example API call to backend FastAPI endpoint
        // const response = await fetch(`${API_BASE_URL}/predict`, {
        //     method: 'POST',
        //     headers: { 'Content-Type': 'application/json' },
        //     body: JSON.stringify({ region, crop })
        // });
        // const data = await response.json();
        
        // Simulated response delay for demonstration
        setTimeout(() => {
            const mockYield = (Math.random() * (8.5 - 6.0) + 6.0).toFixed(1);
            yieldOutput.innerHTML = `${mockYield} T/ha`;
        }, 800);

    } catch (error) {
        console.error("Error connecting to backend API:", error);
        yieldOutput.innerHTML = "Error connecting";
        alert("Make sure your FastAPI backend server is running locally on port 8000!");
    }
}

// Initialize dashboard components on load
document.addEventListener("DOMContentLoaded", () => {
    console.log("CR-YPP Frontend Dashboard Initialized successfully.");
});