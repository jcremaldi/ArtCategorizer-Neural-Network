import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from keras.applications.resnet50 import preprocess_input
from keras.models import load_model, model_from_json

st.sidebar.header('ArtCategorizer')

@st.cache(allow_output_mutation=True)
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
    mov_option = st.sidebar.selectbox('Art movement: ',('Cubism', 'Impressionism', 'Renaissance', 'Surrealism'))    
    if mov_option == 'Cubism':
        image_num = st.sidebar.slider('Image number:', 1, 10, 1)
        art_im = [766,1396,2933,3159,4104,9354,10204,11156,38281,90806]
        path = ''.join(['data/validation/Cubism/',str(art_im[image_num-1]),'.jpg'])    
    if mov_option == 'Impressionism':
        image_num = st.sidebar.slider('Image number:', 1, 10, 1)
        art_im = [78820,88929,79600,89945,91114,89497,89497,88728,86708,79694]
        path = ''.join(['data/validation/Impressionism/',str(art_im[image_num-1]),'.jpg'])
    if mov_option == 'Renaissance':
        image_num = st.sidebar.slider('Image number:', 1, 10, 1)
        art_im = [6672,8053,11574,18613,17657,18039,18807,19157,20470,10982]
        path = ''.join(['data/validation/Renaissance/',str(art_im[image_num-1]),'.jpg'])
    if mov_option == 'Surrealism':
        image_num = st.sidebar.slider('Image number:', 1, 10, 1)
        art_im = [12042,14634,17306,18372,21897,21500,11030,17896,21755,24870]
        path = ''.join(['data/validation/Surrealism/',str(art_im[image_num-1]),'.jpg'])
            
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

labels = ['Cubism', 'Impressionism', 'Renaissance', 'Surrealism']
sizes = [pred_probs[0,0], pred_probs[0,1], pred_probs[0,2], pred_probs[0,3]]

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',startangle=90)
ax1.axis('equal')

st.pyplot()
st.image(img, caption=None, width = 448, use_column_width=False, clamp=False, channels='RGB', format='JPEG')