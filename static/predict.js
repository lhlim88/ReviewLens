// ReviewLens - "try it yourself" live prediction

async function checkSentiment() {
  const text = document.getElementById("reviewInput").value;
  const resultDiv = document.getElementById("predictionResult");

  resultDiv.textContent = "Checking...";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text })
    });
    const data = await response.json();
    resultDiv.textContent = `Sentiment: ${data.sentiment} (confidence: ${data.confidence})`;
  } catch (err) {
    resultDiv.textContent = "Error getting prediction.";
    console.error(err);
  }
}
