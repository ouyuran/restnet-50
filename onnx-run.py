import onnxruntime as ort
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor

# 1. 加载 processor（和训练时一致）
processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")

# 2. 加载 ONNX 模型
session = ort.InferenceSession("onnx_resnet/model.onnx")

# 3. 读取图片
image = Image.open("cat.png")  # 换成你的图片路径

# 4. 预处理（⚠️ 必须和原模型一致）
inputs = processor(image, return_tensors="np")

# 5. ONNX 推理
outputs = session.run(None, dict(inputs))

# 6. 解析结果
logits = outputs[0]
predicted_class_id = np.argmax(logits, axis=-1)[0]

# 7. 转 label
from transformers import ResNetForImageClassification
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

print("预测类别：", model.config.id2label[predicted_class_id])