function classifyImage() {
    const fileInput = document.getElementById('imageUpload');
    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('image', file);

    fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // Handle the classification result
        document.getElementById('result').innerText = 'Classified as: ' + data.class;
    })
    .catch(error => {
        console.error('Error:', error);
    });
}