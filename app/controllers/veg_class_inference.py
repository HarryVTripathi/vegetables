import io
import traceback
import os
from pathlib import Path
import numpy as np
import openvino.runtime as ov
from PIL import Image
from fastapi import APIRouter
from fastapi import FastAPI, File, UploadFile, Response


MEAN = 255 * np.array([0.485, 0.456, 0.406])
STD = 255 * np.array([0.229, 0.224, 0.225])
VEG_CLASSES = ['Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber', 'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato']


model_xml = str(Path(__file__).parents[2]) + "/models/resnet18_int8.xml"
core = ov.Core()
model = core.read_model(model=model_xml)
compiled_model = core.compile_model(model=model, device_name="CPU")
veg_class_router = APIRouter(tags=["classification"])


@veg_class_router.post("/veg-label")
def get_veg_label(file: UploadFile = File()):
    try:
        im = Image.open(file.file)
        im_path = os.path.join(__file__.split("veg_class_inference.py")[0], "file.jpg")

        if im.mode in ("RGBA", "P"): 
            im = im.convert("RGB")
        im.save(im_path, 'JPEG', quality=50)
    
        img = Image.open(im_path)
        print(type(img))

        x = np.array(img)
        x = x.transpose(-1, 0, 1)
        x = (x - MEAN[:, None, None]) / STD[:, None, None]
        x = np.expand_dims(x, 0)

        output_layer = compiled_model.output(0)
        result_infer = compiled_model(x)[output_layer]
        result_index = int(np.argmax(result_infer))
        return {"result": VEG_CLASSES[result_index], "filename": file.filename}
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        return Response(status_code=400)
