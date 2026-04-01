import numpy as np
import onnxruntime as ort

session = ort.InferenceSession("onnx_resnet/model.onnx")

x = np.load("cat.npy")

outputs = session.run(None, {"pixel_values": x})

logits = outputs[0]
predicted_class_id = np.argmax(logits, axis=-1)[0]

from transformers import ResNetForImageClassification
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

print("预测类别：", model.config.id2label[predicted_class_id])