import requests
from bs4 import BeautifulSoup
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input, decode_predictions


class WebScraper:
    def __init__(self, url):
        self.url = url

    def scrape_images(self, save_dir):
        try:
            # Make a GET request to the URL
            response = requests.get(self.url)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            # Find all image tags in the HTML
            image_tags = soup.find_all("img")

            os.makedirs(save_dir, exist_ok=True)

            # Download and save each image
            for i, image_tag in enumerate(image_tags):
                image_url = image_tag["src"]
                image_file = f"image{i}.jpg"
                image_path = os.path.join(save_dir, image_file)

                image_response = requests.get(image_url)
                image_response.raise_for_status()

                with open(image_path, "wb") as f:
                    f.write(image_response.content)

            print("Images successfully scraped and saved!")

        except requests.exceptions.RequestException as e:
            print(f"Failed to scrape images. Error: {str(e)}")


class ImageClassifier:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def classify_image(self, image_path):
        try:
            img = image.load_img(image_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = preprocess_input(x)
            x = tf.expand_dims(x, axis=0)

            prediction = self.model.predict(x)

            predicted_label = decode_predictions(prediction, top=1)[0][0][1]

            return predicted_label

        except Exception as e:
            print(f"Failed to classify image. Error: {str(e)}")


class WebImageClassification:
    def __init__(self, url, save_dir, model_path):
        self.url = url
        self.save_dir = save_dir
        self.model_path = model_path

    def run(self):
        web_scraper = WebScraper(self.url)
        web_scraper.scrape_images(self.save_dir)

        image_classifier = ImageClassifier(self.model_path)
        image_files = os.listdir(self.save_dir)

        for image_file in image_files:
            image_path = os.path.join(self.save_dir, image_file)
            predicted_label = image_classifier.classify_image(image_path)
            print(f"{image_file}: {predicted_label}")


def main():
    url = "https://example.com"
    save_dir = "image_data"
    model_path = "model.h5"

    web_image_classification = WebImageClassification(
        url, save_dir, model_path)
    web_image_classification.run()


if __name__ == "__main__":
    main()
