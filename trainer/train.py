import matplotlib.pyplot as plt
import numpy as np
import onnx
import tensorflow as tf
import tf2onnx

print(tf.__version__)

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    logical_gpus = tf.config.experimental.list_logical_devices("GPU")
    print("Physical GPUs: {}, Logical GPUs: {}".format(len(gpus), len(logical_gpus)))
else:
    print("CPU only")

x = np.arange(-10, 10, 0.0001)
tx = x / 10.0
y = 0.8 * x**2 + 0.2

model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(100, activation=None),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Dense(1, activation=None),
    ]
)
model.compile("sgd", "mse")
model.build(input_shape=(0, 1))
model.summary()
model.optimizer.learning_rate = 0.0001
model.fit(tx, y, epochs=5)

# 予測結果を取得
y_pred = model.predict(tx)

# モデルをONNX形式に変換
onnx_model, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=[tf.TensorSpec([None, 1], tf.float32, name="input")],
    opset=13,
)

# ONNX形式でモデルを保存
onnx.save(onnx_model, "../predictor/model.onnx")


def check_dump(x):
    print(0.8 * x**2 + 0.2, model.predict([x / 10.0]))


check_dump(3)
check_dump(8)
check_dump(20)

# グラフの描画
plt.figure(figsize=(10, 6))
plt.scatter(tx, y, label="Actual Data", alpha=0.5, s=1)
plt.plot(tx, y_pred, "r-", label="Predicted", linewidth=2)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Actual vs Predicted")
plt.grid(True)
plt.legend()
plt.savefig("plot.png", dpi=300, bbox_inches="tight")
plt.close()
