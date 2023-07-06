let model;
const modelURL = 'https://teachablemachine.withgoogle.com/models/_ZS4aE8h-/model.json';
const predictButton = document.getElementById('predict');
const uploadButton = document.getElementById('upload');
const resultsDiv = document.getElementById('results');

async function loadModel() {
  console.log('Loading model...');
  model = await tf.loadLayersModel(modelURL);
  console.log('Model loaded.');
}

loadModel();

uploadButton.addEventListener('change', (e) => {
  let reader = new FileReader();
  reader.onload = function(event) {
    let img = new Image();
    img.onload = function() {
      let tensor = tf.browser.fromPixels(img)
        .resizeNearestNeighbor([224, 224])
        .toFloat()
        .expandDims();

      predict(tensor);
    };
    img.src = event.target.result;
  };
  reader.readAsDataURL(e.target.files[0]);
}, false);

async function predict(tensor) {
  let prediction = await model.predict(tensor).data();
  let topPrediction = prediction.indexOf(Math.max(...prediction));

  resultsDiv.innerText = `Prediction: ${topPrediction}`;
}

predictButton.addEventListener('click', () => {
  uploadButton.click();
});
