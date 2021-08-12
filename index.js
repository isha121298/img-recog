
let net;
const webcamElement = document.getElementById('webcam');
const classifier = knnClassifier.create();
/* creating arrays for each class
var Isha_set=localStorage.getItem("isha")? localStorage.getItem("isha").split("ISHA"):[];
var is_new = Isha_set.map(Ish =>{
  return JSON.parse(Ish);

})
const Prabhat_set=[];
const Shourya_set=[];
const Avi_set=[];  */

async function app() {
 console.log('Loading mobilenet..');

  // Load the model.
  net = await mobilenet.load();
  console.log('Successfully loaded model');

  

  // Create an object from Tensorflow.js data API which could capture image 
  // from the web camera as Tensor.
  const webcam = await tf.data.webcam(webcamElement);

  // Reads an image from the webcam and associates it with a specific class
  // index.
  const addExample = async classId => {
    // Capture an image from the web camera.
    const img = await webcam.capture();

    // Get the intermediate activation of MobileNet 'conv_preds' and pass that
    // to the KNN classifier.
    const activation = net.infer(img, true);

    // Pass the intermediate activation to the classifier.
    classifier.addExample(activation, classId);
    /*if(classId==0){
      Isha_set.push(img);
      var nm1=Isha_set.map(val=>{
        return JSON.stringify(val)
      })
      var tm1=nm1.join("ISHA")
      localStorage.setItem("isha",tm1)
    }
    else if(classId==1){
      Prabhat_set.push(img);
    }
    else if(classId==2){
      Shourya_set.push(img);
    }
    else if(classId==3){
      Avi_set.push(img);
    } */

    //console.log(Isha_set);
    //is.show();
    
    //await img.save('localStorage://mymodel');
  
    //classifier.getClassifierDataset();
    // Dispose the tensor to release the memory.
    img.dispose();
    };

  // When clicking a button, add an example for that class.
  document.getElementById('Isha').addEventListener('click', () => addExample(0));
  document.getElementById('Prabhat').addEventListener('click', () => addExample(1));
  document.getElementById('Shourya').addEventListener('click', () => addExample(2));
  document.getElementById('Avi').addEventListener('click', () => addExample(3));

  while (true) {
    if (classifier.getNumClasses() > 0) {
      const img = await webcam.capture();

      // Get the activation from mobilenet from the webcam.
      const activation = net.infer(img, 'conv_preds');
      // Get the most likely class and confidence from the classifier module.
      const result = await classifier.predictClass(activation);

      const classes = ['Isha', 'Prabhat', 'Shourya','Avi'];
      document.getElementById('console').innerText = `
        prediction: ${classes[result.label]}\n
        probability: ${result.confidences[result.label]}`;

      // Dispose the tensor to release the memory.
      img.dispose();
    }

    await tf.nextFrame();
  }
}
app();