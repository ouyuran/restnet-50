from optimum.onnxruntime import ORTModelForImageClassification

ort_model = ORTModelForImageClassification.from_pretrained(
    "microsoft/resnet-50",
    export=True
)

ort_model.save_pretrained("./onnx_resnet")