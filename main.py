import tensorflow as tf
from datasets import load_dataset

from models import build_ctc_model
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


class CTCLossModel(tf.keras.Model):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

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

        print(f"Epoch {epoch + 1} completed. Avg Loss: {logs['loss']:.4f}")


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
dataset = load_dataset("speech-uk/voice-of-america", split='train', streaming=True)

prepared_ds = prepare_dataset(dataset)
base_model = build_ctc_model(num_classes=len(VOCAB))
ctc_model = CTCLossModel(base_model)
train(ctc_model, prepared_ds, epochs=1)
