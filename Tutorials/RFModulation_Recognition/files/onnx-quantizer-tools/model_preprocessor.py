from onnxruntime.quantization.shape_inference import quant_pre_process

def model_preprocessor(onnx_fp32_path):
    quant_pre_process(onnx_fp32_path, 
                      output_model_path='temporary_optimized_model.onnx')