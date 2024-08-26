
import numpy as np
from PIL import ImageOps, Image

def classify(image, model, class_names):
    # Resize image to (224, 224) as required by ResNet
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)

    # Convert image to numpy array
    image_array = np.asarray(image)

    # Normalize image to range [0, 1]
    # normalized_image_array = image_array.astype(np.float32) / 255.0
    normalized_image_array = image_array.astype(np.float32)



    # Reshape and add batch dimension
    data = np.expand_dims(normalized_image_array, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(data)
    index = np.argmax(prediction)

    # Get the class name and confidence score
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name, confidence_score
