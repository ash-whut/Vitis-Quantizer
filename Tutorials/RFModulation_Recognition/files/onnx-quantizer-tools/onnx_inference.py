import numpy as np

def predict(session, X_test):
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    y_pred = []
    for x in X_test:
        input_data = {input_name: np.expand_dims(x.astype(np.float32), axis=0)}
        output = session.run([output_name], input_data)[0]
        y_pred.append(output)

    return np.array(y_pred)