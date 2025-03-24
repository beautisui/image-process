import * as cocoSsd from "@tensorflow-models/coco-ssd";
import * as tf from "@tensorflow/tfjs";
import fs from "fs";
import sharp from "sharp";

const createImageTensor = async (imagePath) => {
  const imageBuffer = fs.readFileSync(imagePath);
  const { data, info } = await sharp(imageBuffer)
    .toFormat("raw")
    .toBuffer({ resolveWithObject: true });  

  return tf.tensor3d(data, [info.height, info.width, 3], "int32");
};

async function runObjectDetection(imagePath) {
  console.log("Loading model...");
  const model = await cocoSsd.load();
  console.log("Model loaded!");

  const decodedImage = await createImageTensor(imagePath);
  console.log(decodedImage);

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
