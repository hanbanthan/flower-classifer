import argparse
import json
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt

def load_class_names(file_path):
    with open(file_path, "r") as f:
        return json.load(f)
    
import tensorflow as tf
from PIL import Image
import numpy as np

def process_image(image_path):
    
    image = Image.open(image_path).convert('RGB')
    
    image = np.asarray(image)
    
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    
    image = tf.image.resize(image, (224, 224))
    
    image = image / 255.0
    
    image = tf.expand_dims(image, axis=0)
    
    return image

def predict(image_path, model, class_names, top_k=5):
    
    img = process_image(image_path)
    
    probs = model.predict(img)
    
    top_k_preds = probs[0].argsort()[-top_k:][::-1]
    
    class_labels = [class_names[str(int(i)+1)] for i in top_k_preds]
    
    return probs[0][top_k_preds], class_labels
    
def main():
    parser = argparse.ArgumentParser(description="Predict the class of a flower image")
    parser.add_argument('image_path', type=str, help='Path to the image to be predicted')
    parser.add_argument('model_path', type=str, help='Path to the trained model')
    parser.add_argument('--top_k', type=int, default=5, help='Return the top K most likely classes')
    parser.add_argument('--category_names', type=str, default='flower_data/cat_to_name.json', help='Path to the category labels file')
    
    args = parser.parse_args()
    
    model = load_model(args.model_path, custom_objects={'KerasLayer': hub.KerasLayer})
    class_names = load_class_names(args.category_names)
    
    probs, classes = predict(args.image_path, model, class_names, top_k=args.top_k)
    
    print(f"Top {args.top_k} Predictions:")
    
    for i in range(args.top_k):
        print(f"{classes[i]}: {probs[i]:.4f}")
        
    image = Image.open(args.image_path)
    plt.imshow(image)
    plt.title(f"Predicted: {classes[0]} ({probs[0]:.4f})")
    plt.show()
    
if __name__ == '__main__':
    main()