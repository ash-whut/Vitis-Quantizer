from onnxruntime.quantization import quantize_dynamic, QuantType, QuantFormat
from model_preprocessor import model_preprocessor
import argparse 
import os

def dynamic_quantizer(input_path, output_path, per_channel_quantization):
    model_preprocessor(input_path)
    
    model_fp32 = 'temporary_optimized_model.onnx'
    model_quant = output_path
    
    quantize_dynamic(
        model_input=model_fp32,
        model_output=model_quant,
        per_channel=per_channel_quantization,
        weight_type=QuantType.QUInt8  # Quantize weights to int8
    )

    os.remove(model_fp32)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ONNX Dynamic Quantizer - UINT8 weights"
    )
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--pc", required=False, type=bool, default=False)
    args = parser.parse_args()

    dynamic_quantizer(args.input, args.output, args.pc)