from onnxruntime.quantization import quantize_static, QuantType, QuantFormat
from model_preprocessor import model_preprocessor
from data_loader import data_loader
from calibration_data_reader import MyCalibrationDataReader
import argparse
import os
from sklearn.model_selection import train_test_split

def static_quantizer(input_path, output_path, per_channel_quantization, qop):

    X_train = data_loader()['X_train']

    model_preprocessor(input_path)
    optimized_model_path_temp = 'temporary_optimized_model.onnx'
    
    quantize_static(
    model_input=optimized_model_path_temp,
    model_output=output_path,
    calibration_data_reader=MyCalibrationDataReader('input', X_train[:100]),
    quant_format=QuantFormat.QOperator if qop else QuantFormat.QDQ,
    per_channel=per_channel_quantization,
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QInt8,
    # extra_options={
    #     "ActivationSymmetric": True,  # Enforce symmetric quantization for activations
    #     "WeightSymmetric": True       # Enforce symmetric quantization for weights
    # }
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ONNX Static Quantizer - INT8 weights"
    )
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--pc", required=False, type=bool, default=False)
    parser.add_argument("--qop", required=False, type=bool, default=False)
    args = parser.parse_args()

    static_quantizer(args.input, args.output, args.pc, args.qop)