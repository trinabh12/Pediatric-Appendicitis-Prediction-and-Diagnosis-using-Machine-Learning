const form = document.getElementById("predictionForm");
const resultBox = document.getElementById("result");
const diagnosisEl = document.getElementById("diagnosis");
const confidenceEl = document.getElementById("confidence");
const gallery = document.getElementById("gallery");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const formData = new FormData(form);

  const res = await fetch("/predict", {
    method: "POST",
    body: formData,
  });

  const data = await res.json();
  resultBox.classList.remove("hidden");
  diagnosisEl.textContent = `Diagnosis: ${data.diagnosis}`;
  confidenceEl.textContent = data.diagnosis === "Appendicitis"
    ? `Risk Score: ${data.risk_score.toFixed(2)}`
    : `Confidence: ${data.confidence}%`;
});

const imageInput = document.getElementById("ultrasound_image");
imageInput.addEventListener("change", () => {
  gallery.innerHTML = "";
  Array.from(imageInput.files).forEach((file) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = document.createElement("img");
      img.src = e.target.result;
      img.alt = file.name;
      gallery.appendChild(img);
    };
    reader.readAsDataURL(file);
  });
});
