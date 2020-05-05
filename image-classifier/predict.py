import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import argparse
import PIL
import json

parse = argparse.ArgumentParser()

parse.add_argument ('--image_path', help = 'Path to the image file', type = str)
parse.add_argument('--model', help='Path to the saved model', type=str)
parse.add_argument ('--top_k', default = 5, help = 'Top K most likely classes where K is an Integer', type = int)
parse.add_argument ('--category_names' , default = 'label_map.json', help = 'Mapping of categories to real names file, in JSON format', type = str)

args = parse.parse_args()

def process_image(img):
    image = np.squeeze(img)
    image = tf.image.resize(image, [224, 224])
    image /= 255
    return image

def predict(image_path, model, top_k, category_names):
    processed_image = process_image(image_path)
    prediction = model.predict(np.expand_dims(processed_image, axis=0))
    top_values, top_indices = tf.math.top_k(prediction, top_k)
    print("The probabilities of top ", top_k, "is", top_values.numpy()[0])
    top_classes = [class_names[str(top_value+1)] for top_value in top_indices.numpy()[0]]
    print("The classes of top ", top_k, "is", top_classes)
    return top_values.numpy()[0], top_classes

if __name__ == "__main__":
    model = tf.keras.models.load_model(str(args.model), custom_objects={'KerasLayer':hub.KerasLayer})
    image = np.asarray(PIL.Image.open(args.image_path))
    top_k = args.top_k
    category_names = args.category_names
    with open(str(args.category_names), 'r') as f:
        class_names = json.load(f)
        
    print(predict(image, model, top_k, category_names))
    
    