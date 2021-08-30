import torch
import numpy as np
from torchvision import transforms
from train_model import data
from PIL import Image, ImageOps
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
import io
import re
from torchsummary import summary
from train_model import Flatten


model = torch.load('./full_model_CNN.pt')

class Prediction:
    def __init__(self):
        self.model = model

    def predict_digit(self, digit):
        image_data = re.sub('^data:image/.+;base64,', '', digit)
        base64_decoded = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(base64_decoded)).resize((28, 28))
        corrected_image = Image.new("RGBA", image.size, "WHITE")
        corrected_image.paste(image, mask=image)
        image = corrected_image.convert('L')
        inverted_image = ImageOps.invert(image)
        pil_to_tensor = transforms.ToTensor()(inverted_image).unsqueeze_(0)
        img = pil_to_tensor.view(1, 1, 28, 28)
        with torch.no_grad():
            logps = self.model(img)
        ps = torch.exp(logps)
        probab = list(ps.numpy()[0])
        pred_label = probab.index(max(probab))
        img_base64 = self.view_classify(image, ps)
        return pred_label, img_base64

    def predict_digit_from_db(self):
        from main import Digit, db
        db.create_all()
        digits = Digit.query.all()
        for digit in digits:
            image_data = re.sub('^data:image/.+;base64,', '', digit.digit_image)
            base64_decoded = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(base64_decoded)).resize((28, 28))
            new_image = Image.new("RGBA", image.size, "WHITE")
            new_image.paste(image, mask=image)
            image = new_image.convert('L')
            image = ImageOps.invert(image)
            pil_to_tensor = transforms.ToTensor()(image).unsqueeze_(0)
            img = pil_to_tensor.view(1, 1, 28, 28)
            with torch.no_grad():
                logps = self.model(img)

            ps = torch.exp(logps)
            probab = list(ps.numpy()[0])
            pred_label = probab.index(max(probab))
            print(pred_label)


    def test_model(self):
        summary(self.model, (1, 28, 28))
        correct_count, all_count = 0, 0
        for images, labels in data.valloader:
            for i in range(len(labels)):
                img = images[i].view(1, 1, 28, 28)
                with torch.no_grad():
                    logps = self.model(img)

                ps = torch.exp(logps)
                probab = list(ps.numpy()[0])
                pred_label = probab.index(max(probab))
                true_label = labels.numpy()[i]
                if (true_label == pred_label):
                    correct_count += 1
                all_count += 1

        print("Number Of Images Tested =", all_count)
        print("\nModel Accuracy =", (correct_count / all_count))


    def view_classify(self, img, ps):
        ''' Function for viewing an image and it's predicted classes.
        '''
        ps = ps.data.numpy().squeeze()
        fig, (ax1, ax2) = plt.subplots(figsize=(6, 3), ncols=2)
        ax1.imshow(img)
        ax1.axis('off')
        ax2.barh(np.arange(10), ps)
        ax2.set_aspect(0.1)
        ax2.set_yticks(np.arange(10))
        ax2.set_yticklabels(np.arange(10))
        ax2.set_title('Class Probability')
        ax2.set_xlim(0, 1.1)
        plt.tight_layout()
        my_stringIObytes = io.BytesIO()
        plt.savefig(my_stringIObytes, format='png')
        my_stringIObytes.seek(0)
        my_base64_jpgData = base64.b64encode(my_stringIObytes.read()).decode("utf-8")
        return my_base64_jpgData
