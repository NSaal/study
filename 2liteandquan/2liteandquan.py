import tensorflow as tf
import sys
import os
if sys.version_info.major>=3:
    import pathlib
else:
    import pathlib2 as pathlib

#def quantization():
saved_model_dir="C:\\Users\\G\\source\\repos\\study\\2liteandquan\\save\\tfmodel2\\"
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()
tflite_model_dir=pathlib.Path("C:\\Users\\G\\source\\repos\\study\\2liteandquan\\save")
#tflite_model_dir.mkdir(exit_ok=True,parents=True)
tflite_model_file=pathlib.Path("C:\\Users\\G\\source\\repos\\study\\2liteandquan\\save\\2lite.tflite")
tflite_model_file.write_bytes(tflite_model)

converter.optimizations=[tf.lite.Optimize.DEFAULT]
minst_train,_=tf.keras.datasets.mnist.load_data()
images=tf.cast(minst_train[0],tf.float32)/255.0
minst_ds=tf.data.Dataset.from_tensor_slices((images)).batch(1)
def representive_data_gen():
    for input_value in minst_ds.take(100):
        yield [input_value]

converter.representive_dataset=representive_data_gen

tflite_model_quant=converter.convert()
tflite_model_quant_file=pathlib.Path("C:\\Users\\G\\source\\repos\\study\\2liteandquan\\save\\quant.tflite")
tflite_model_quant_file.write_bytes(tflite_model_quant)
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
tflite_model_quant.evaluate(x_test,  y_test, verbose=2)