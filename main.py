import tensorflow as tf
# from tensorflow import keras
import keras
from datasets import load_dataset

from models import build_ctc_model, build_ctc_model_CNN
from data_processing import prepare_dataset, VOCAB


def ctc_batch_cost(y_true, y_pred, input_lengths, label_lengths):
    """
    y_true: int32 tensor of shape (batch_size, max_label_length)
    y_pred: float32 tensor of shape (batch_size, max_time, num_classes + 1) — logits
    input_lengths: int32 tensor of shape (batch_size,) — actual input sequence lengths
    label_lengths: int32 tensor of shape (batch_size,) — actual label lengths
    """
    indices = tf.where(tf.not_equal(y_true, -1))
    values = tf.gather_nd(y_true, indices)
    shape = tf.shape(y_true, out_type=tf.int64)

    sparse_labels = tf.SparseTensor(indices=indices, values=values, dense_shape=tf.cast(shape, tf.int64))

    y_pred = tf.transpose(y_pred, [1, 0, 2])

    loss = tf.nn.ctc_loss(
        labels=sparse_labels,
        logits=y_pred,
        label_length=None,
        logit_length=input_lengths,
        blank_index=-1
    )

    return loss


@keras.saving.register_keras_serializable()
class CTCLossModel(tf.keras.Model):
    def __init__(self, base_model, **kwargs):
        super().__init__(**kwargs)
        self.base = base_model
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    def get_config(self):
        config = super().get_config()
        # Serialize the inner model by config, not the object
        config.update({
            "base_model_config": self.base.get_config(),
            "base_model_class": self.base.__class__.__name__
        })
        return config

    @classmethod
    def from_config(cls, config):
        base_model_config = config.pop("base_model_config")
        base_model_class_name = config.pop("base_model_class")

        # Rebuild the base model using its class and config
        base_model_class = getattr(tf.keras.models, base_model_class_name, None)
        if base_model_class is None:
            base_model = tf.keras.Model.from_config(base_model_config)
        else:
            base_model = base_model_class.from_config(base_model_config)

        return cls(base_model, **config)

    def get_build_config(self):
        # Keras saves this at save-time to later rebuild layers
        return {"input_shape": (None, None, 64)}

    def build_from_config(self, config):
        # Called during loading to rebuild the model's variables
        input_shape = config["input_shape"]
        self.build(input_shape)

    @property
    def metrics(self):
        return [self.loss_tracker]

    def train_step(self, data):
        features = data["features"]
        labels = data["label"]
        input_lengths = data["input_length"]
        label_lengths = data["label_length"]

        with tf.GradientTape() as tape:
            logits = self.base(features, training=True)
            input_lengths = tf.fill([tf.shape(logits)[0]], tf.shape(logits)[1])
            loss = ctc_batch_cost(labels, logits, input_lengths, label_lengths)

        grads = tape.gradient(loss, self.base.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.base.trainable_variables))

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def call(self, inputs):
        return self.base(inputs)


def train(model: CTCLossModel, dataset: tf.data.Dataset, epochs: int):
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        model.reset_metrics()

        for step, batch in enumerate(dataset):
            logs = model.train_step(batch)
            if step % 10 == 0:
                print(f"  Step {step}: Loss = {logs['loss']:.4f}")
                if logs['loss'] < 5:
                    dummy_input = tf.random.normal([1, 100, 64])
                    _ = model(dummy_input)
                    model.save('./final_model.keras')
                    return
        print(f"Epoch {epoch + 1} completed. Avg Loss: {logs['loss']:.4f}")
    dummy_input = tf.random.normal([1, 100, 64])
    _ = model(dummy_input)
    model.save('./final_model.keras')


"""
Dataset element structure:
{
    audio: {
        path: str,
        array: np.ndarray,
        sampling_rate: int,
    },
    duration: float,
    transcription: str,
}
"""
# dataset = load_dataset("speech-uk/voice-of-america", split='train', streaming=True)

# prepared_ds = prepare_dataset(dataset)
base_model = build_ctc_model(num_classes=len(VOCAB))
ctc_model = CTCLossModel(base_model)
# train(ctc_model, prepared_ds, epochs=2)
base_model.summary()
# loaded_model = keras.models.load_model('./final_model.keras', custom_objects={"CTCLossModel": CTCLossModel})
