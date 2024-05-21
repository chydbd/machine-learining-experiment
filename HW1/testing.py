import torch
from torch import nn
from models import test_image_read,Model
from matplotlib import pyplot as plt
import gradio as gr
import json

model = Model()

model_path = 'model.pth'  # 模型文件的路径
model.load_state_dict(torch.load(model_path))

def model_predict(file_path):
    data = test_image_read(file_path)
    model.eval()
    with torch.no_grad():
        outputs = model(data)
    type_index = torch.argmax(outputs)
    if type_index == 0:
        output_type = "Angry"
    elif type_index == 1:
        output_type = "Fear"
    elif type_index == 2:
        output_type = "Happy"
    elif type_index == 3:
        output_type = "Neutral"
    elif type_index == 4:
        output_type = "Sad"
    elif type_index == 5:
        output_type = "Surprise"
    return output_type

iface = gr.Interface(fn = model_predict,inputs=gr.File(label="File",type="filepath"),outputs="text")
iface.launch()
