import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras.applications.resnet50 import preprocess_input
from keras.models import load_model, model_from_json


#------------------------------------------------------------------------------
print('load the model')
# architecture and weights from HDF5
model = load_model('models/keras/model.h5')

# architecture from JSON, weights from HDF5
with open('models/keras/architecture.json') as f:
    model = model_from_json(f.read())
model.load_weights('models/keras/weights.h5')
#------------------------------------------------------------------------------

print('predict')
validation_img_paths = ["data/validation/Impressionism/88929.jpg",
                        "data/validation/Romanticism/46869.jpg"]
img_list = [Image.open(img_path) for img_path in validation_img_paths]

img_size = 224
validation_batch = np.stack([preprocess_input(np.array(img.resize((img_size, img_size))))
                             for img in img_list])

pred_probs = model.predict(validation_batch)


fig, axs = plt.subplots(1, len(img_list), figsize=(20, 5))
for i, img in enumerate(img_list):
    ax = axs[i]
    ax.axis('off')
    ax.set_title("{:.0f}% Impressionism, {:.0f}% Realism, {:.0f}% Romanticism".format(100*pred_probs[i,0],
                                                          100*pred_probs[i,1],
                                                          100*pred_probs[i,2]))
    ax.imshow(img)



























