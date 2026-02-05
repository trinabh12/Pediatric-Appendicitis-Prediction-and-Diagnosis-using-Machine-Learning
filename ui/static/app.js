document.getElementById('predictionForm').addEventListener('submit', function(e) {
    e.preventDefault();

    const formData = new FormData(this);

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('diagnosis').textContent = `Diagnosis: ${data.diagnosis}`;
        document.getElementById('confidence').textContent = `Confidence: ${data.confidence}%`;
        document.getElementById('result').classList.remove('hidden');
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while making the prediction.');
    });
});