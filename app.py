import streamlit as st
import os
import requests
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import matplotlib.pyplot as plt
FILE_ID = '1o4j3QmI3cH1cMi_plMrk7Sp7yFY6oOIr'
st.set_page_config(page_title="RTG Analysis", layout="wide")
st.title('🩺Grad-CAM for lung X-ray scans')
st.write("An app to support doctors in detecting pneumonia using deep learning.")


#fja za grad cam
def generate_gradcam(model, input_batch, target_layer):
    res={"grads": None, "acts": None}
    def save_grads(module, grad_input, grad_output): 
        grad = grad_output[0].clone().detach()
        res["grads"] = grad

    def save_acts(module, input, output): 
        res["acts"] = output.detach().clone()

    h1=target_layer.register_forward_hook(save_acts)
    h2=target_layer.register_full_backward_hook(save_grads)

    model.eval()
    input_batch.requires_grad=True

    with torch.enable_grad():
        output=model(input_batch)
        target_class=output.argmax(dim=1).item()
        model.zero_grad()
        #output[0, target_class].backward()
        loss = output[0, target_class]
        loss.backward(retain_graph=False)

    h1.remove(); h2.remove()

    #izrada toplinske karte
    grads=res["grads"].cpu().data.numpy()
    acts=res["acts"].cpu().data.numpy()
    weights=np.mean(grads, axis=(2,3))[0]
    cam=np.zeros(acts.shape[2:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam+=w*acts[0, i, :, :]

    cam=np.maximum(cam, 0)
    cam=cv2.resize(cam, (224, 224))
    if cam.max()>0:
        cam=cam/cam.max()
    return cam, target_class

#ucitavanje modela
@st.cache_resource
def load_my_model():
    url = 'https://huggingface.co/ivonapranjic/pneumonia-vgg16-model/resolve/main/pneumonia_vgg16.pth'
    output = 'pneumonia_vgg16.pth'
    
    
    if not os.path.exists(output):
        with st.spinner('Preuzimanje modela s Hugging Face-a... Molimo pričekajte.'):
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status() # Provjerava je li link ispravan
                with open(output, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            except Exception as e:
                st.error(f"Greška pri preuzimanju modela: {e}")
                return None

    
    model = models.vgg16(weights=None)
    
    # Prilagodba klasifikatora 
    n_inputs = model.classifier[6].in_features
    model.classifier[6] = nn.Sequential(
        nn.Linear(n_inputs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 2)
    )
    
    
    model.load_state_dict(torch.load(output, map_location=torch.device('cpu')))
    model.eval()
    
    return model


#glavni dio sučelja:
uploaded_file=st.file_uploader("Upload an X-ray image (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image=Image.open(uploaded_file).convert('RGB')
    st.sidebar.header("Crop")
    width, height = image.size
    
    
    top = st.sidebar.slider("Crop Top (%)", 0, 50, 15)
    bottom = st.sidebar.slider("Crop Bottom (%)", 0, 50, 10)
    left = st.sidebar.slider("Crop Left (%)", 0, 50, 10)
    right = st.sidebar.slider("Crop Right (%)", 0, 50, 10)
    
    
    left_px = (left / 100) * width
    top_px = (top / 100) * height
    right_px = width - (right / 100) * width
    bottom_px = height - (bottom / 100) * height
    
   
    image = image.crop((left_px, top_px, right_px, bottom_px))
    st.sidebar.image(image, caption="Focused zone", use_container_width=True)
    
    


    img_array = np.array(image.convert('L'))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img_array)
    image = Image.fromarray(img_clahe).convert('RGB')

    transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                  ])
    input_tensor=transform(image).unsqueeze(0)

    #gumb za analizu
    if st.button('RUN THE ANALYSIS'):
        try:
            model=load_my_model()

            cam, pred_idx=generate_gradcam(model, input_tensor, model.features[30])
            labeli=["NORMAL", "PNEUMONIA"]

            st.subheader(f"Result: {labeli[pred_idx]}")

            col1, col2=st.columns(2)

            with col1:
                st.image(image, caption="Original image", use_container_width=True)

            with col2:
                img_np=np.array(image.resize((224, 224)))
                heatmap=cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
                heatmap=cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                overlay=cv2.addWeighted(img_np, 0.3, heatmap, 0.7, 0)

                st.image(overlay, caption="Grad-CAM visualization", use_container_width=True)
        except Exception as e:
            st.error(f"The model is not ready yet or the file is missing: {e}")
