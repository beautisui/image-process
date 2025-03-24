import * as cocoSsd from "@tensorflow-models/coco-ssd";
import * as tf from "@tensorflow/tfjs-node";
import fs from "fs";

async function runObjectDetection(imagePath) {
  console.log("Loading model...");
  const model = await cocoSsd.load();
  console.log("Model loaded!");

  const imageBuffer = fs.readFileSync(imagePath);
  const decodedImage = tf.node.decodeImage(imageBuffer);

  console.log("Detecting objects...");
  const predictions = await model.detect(decodedImage);

  console.log("Detected Objects:");
  predictions.forEach((pred, i) => {
    console.log(
      `${i + 1}. ${pred.class} - Confidence: ${(pred.score * 100).toFixed(2)}%`
    );
  });

  decodedImage.dispose();
}

const imagePath = process.argv[2];
if (!imagePath) {
  console.error("Please provide an image path: node recognize.js image.jpg");
  process.exit(1);
}

runObjectDetection(imagePath);
