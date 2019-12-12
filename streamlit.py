import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from keras.applications.resnet50 import preprocess_input
from keras.models import load_model, model_from_json

st.sidebar.header('ArtCategorizer')

def add_model():
    model = load_model('models/keras/model.h5')
    return model
def add_weights(model):
    model.load_weights('models/keras/weights.h5')
    return model

st.sidebar.text('Loading data...')
model = add_model()
with open('models/keras/architecture.json') as f:
    model = model_from_json(f.read())
model = add_weights(model)
st.sidebar.text('Loading Done!')

path_option = st.sidebar.selectbox('Type of path: ',('Validation files', 'Local'))

if path_option == 'Local':
    path = st.sidebar.text_input('File Path: ', 'data/validation/Impressionism/78820.jpg')
elif path_option == 'Validation files':
    mov_option = st.sidebar.selectbox('Art movement: ',('Impressionism', 'Realism', 'Romanticism'))
    if mov_option == 'Impressionism':
        image_num = st.sidebar.slider('Image number:', 1, 10, 1)
        if image_num == 1:
            path = 'data/validation/Impressionism/78820.jpg'
        if image_num == 2:
            path = 'data/validation/Impressionism/88929.jpg'
        if image_num == 3:
            path = 'data/validation/Impressionism/79600.jpg'
        if image_num == 4:
            path = 'data/validation/Impressionism/89945.jpg'
        if image_num == 5:
            path = 'data/validation/Impressionism/91114.jpg'
        if image_num == 6:
            path = 'data/validation/Impressionism/89497.jpg'
        if image_num == 7:
            path = 'data/validation/Impressionism/88728.jpg'
        if image_num == 8:
            path = 'data/validation/Impressionism/86708.jpg'
        if image_num == 9:
            path = 'data/validation/Impressionism/79694.jpg'
        if image_num == 10:
            path = 'data/validation/Impressionism/78370.jpg'
    if mov_option == 'Realism':
        image_num = st.sidebar.slider('Image number:', 1, 10, 1)
        if image_num == 1:
            path = 'data/validation/Realism/94223.jpg'
        if image_num == 2:
            path = 'data/validation/Realism/96385.jpg'
        if image_num == 3:
            path = 'data/validation/Realism/101559.jpg'
        if image_num == 4:
            path = 'data/validation/Realism/96386.jpg'
        if image_num == 5:
            path = 'data/validation/Realism/94158.jpg'
        if image_num == 6:
            path = 'data/validation/Realism/100551.jpg'
        if image_num == 7:
            path = 'data/validation/Realism/101332.jpg'
        if image_num == 8:
            path = 'data/validation/Realism/100719.jpg'
        if image_num == 9:
            path = 'data/validation/Realism/96595.jpg'
        if image_num == 10:
            path = 'data/validation/Realism/95094.jpg'
    if mov_option == 'Romanticism':
        image_num = st.sidebar.slider('Image number:', 1, 10, 1)
        if image_num == 1:
            path = 'data/validation/Romanticism/47172.jpg'
        if image_num == 2:
            path = 'data/validation/Romanticism/48664.jpg'
        if image_num == 3:
            path = 'data/validation/Romanticism/50316.jpg'
        if image_num == 4:
            path = 'data/validation/Romanticism/46049.jpg'
        if image_num == 5:
            path = 'data/validation/Romanticism/49792.jpg'
        if image_num == 6:
            path = 'data/validation/Romanticism/47487.jpg'
        if image_num == 7:
            path = 'data/validation/Romanticism/49004.jpg'
        if image_num == 8:
            path = 'data/validation/Romanticism/50240.jpg'
        if image_num == 9:
            path = 'data/validation/Romanticism/47718.jpg'
        if image_num == 10:
            path = 'data/validation/Romanticism/48010.jpg'

img = Image.open("%s" % (path)) 

validation_img_paths = ["%s" % (path)]
img_list = [Image.open(img_path) for img_path in validation_img_paths]

img_size = 224
validation_batch = np.stack([preprocess_input(np.array(img.resize((img_size, img_size))))
                             for img in img_list])

pred_probs = model.predict(validation_batch)
fig, axs = plt.subplots(1, len(img_list), figsize=(20, 5))

#st.write("{:.0f}% Impressionism, {:.0f}% Realism, {:.0f}% Romanticism".format(100*pred_probs[0,0],
#                                                          100*pred_probs[0,1],
#                                                          100*pred_probs[0,2]))

labels = 'Impressionism', 'Realism', 'Romanticism'
sizes = [pred_probs[0,0], pred_probs[0,1], pred_probs[0,2]]

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',startangle=90)
ax1.axis('equal')

st.pyplot()
st.image(img, caption=None, width = 448, use_column_width=False, clamp=False, channels='RGB', format='JPEG')
