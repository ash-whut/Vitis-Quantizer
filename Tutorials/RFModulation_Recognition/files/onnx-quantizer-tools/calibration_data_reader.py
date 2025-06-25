from onnxruntime.quantization import CalibrationDataReader
import numpy as np 

class MyCalibrationDataReader(CalibrationDataReader):
    def __init__(self, input_name, calibration_data):
        self.input_name = input_name
        self.data = calibration_data
        self.index = 0

    def get_next(self):
        if self.index < len(self.data):
            input_tensor = self.data[self.index].astype('float32')
            input_tensor = np.expand_dims(input_tensor, axis=0)
            self.index += 1
            return {self.input_name: input_tensor}
        return None