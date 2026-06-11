document
  .getElementById("predictionForm")
  .addEventListener("submit", async function (e) {
    e.preventDefault();

    // Collect Input Values
    const data = {
      spread1: parseFloat(document.getElementById("spread1").value),

      PPE: parseFloat(document.getElementById("PPE").value),

      spread2: parseFloat(document.getElementById("spread2").value),

      HNR: parseFloat(document.getElementById("HNR").value),

      MDVP_Shimmer: parseFloat(document.getElementById("MDVP_Shimmer").value),

      voice_amplitude_instability: parseFloat(
        document.getElementById("voice_amplitude_instability").value,
      ),

      MDVP_APQ: parseFloat(document.getElementById("MDVP_APQ").value),

      MDVP_Flo_Hz: parseFloat(document.getElementById("MDVP_Flo_Hz").value),

      Shimmer_APQ5: parseFloat(document.getElementById("Shimmer_APQ5").value),

      D2: parseFloat(document.getElementById("D2").value),

      MDVP_Shimmer_dB: parseFloat(
        document.getElementById("MDVP_Shimmer_dB").value,
      ),

      Shimmer_DDA: parseFloat(document.getElementById("Shimmer_DDA").value),

      Shimmer_APQ3: parseFloat(document.getElementById("Shimmer_APQ3").value),

      MDVP_Jitter_Abs: parseFloat(
        document.getElementById("MDVP_Jitter_Abs").value,
      ),

      MDVP_Fo_Hz: parseFloat(document.getElementById("MDVP_Fo_Hz").value),
    };

    try {
      // API Call
      const response = await fetch(
        "https://parkinson-detection-ai.onrender.com/predict",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(data),
        },
      );

      const result = await response.json();

      // Get Result Elements
      const resultCard = document.getElementById("resultCard");

      const resultTitle = document.getElementById("resultTitle");

      const confidenceText = document.getElementById("confidenceText");

      const confidenceFill = document.getElementById("confidenceFill");

      const recommendationText = document.getElementById("recommendationText");

      // Show Result Card
      resultCard.classList.remove("hidden");

      // Positive Result
      if (result.prediction === 1) {
        resultCard.classList.remove("result-negative");

        resultCard.classList.add("result-positive");

        resultTitle.innerHTML = "⚠ Parkinson Risk Detected";

        recommendationText.innerHTML =
          "Please consult a neurologist for further medical evaluation.";
      }

      // Healthy Result
      else {
        resultCard.classList.remove("result-positive");

        resultCard.classList.add("result-negative");

        resultTitle.innerHTML = "✅ No Strong Parkinson Indicators";

        recommendationText.innerHTML =
          "No strong signs detected. Maintain regular health checkups.";
      }

      confidenceText.innerHTML = `Confidence: ${result.confidence}%`;

      confidenceFill.style.width = `${result.confidence}%`;

      // Scroll to result
      resultCard.scrollIntoView({
        behavior: "smooth",
      });
    } catch (error) {
      alert("Error connecting to backend.");

      console.error(error);
    }
  });
