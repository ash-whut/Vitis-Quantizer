import tensorflow as tf
import tf2onnx
import argparse

def onnx_converter(keras_path, output_path):
    model = tf.keras.models.load_model(keras_path)  # or build your model
    input_signature = [tf.TensorSpec(shape=(None, 1024, 1, 2), dtype=tf.float32, name='input')]
    
    onnx_model, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        opset=13,  # ONNX opset version (13 or higher recommended)
        output_path=output_path
    )    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ONNX Converter for keras models"
    )
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    args = parser.parse_args()

    onnx_converter(args.input, args.output)