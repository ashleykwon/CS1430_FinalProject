import numpy as np
import tensorflow as tf


class FaceDetector:
    def __init__(self, model_path):
        """
        Arguments:
            model_path: a string, path to a pb file.
            visible_device_list: a string.
        """
        with tf.io.gfile.GFile(model_path, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        graph = tf.Graph()
        with graph.as_default():
            tf.import_graph_def(graph_def, name="import")

        self.input_image = graph.get_tensor_by_name("import/image_tensor:0")
        self.output_ops = [
            graph.get_tensor_by_name("import/boxes:0"),
            graph.get_tensor_by_name("import/scores:0"),
            graph.get_tensor_by_name("import/num_boxes:0"),
        ]

        config_proto = tf.compat.v1.ConfigProto(log_device_placement=False)
        self.sess = tf.compat.v1.Session(graph=graph, config=config_proto)

    def __call__(self, image, score_threshold=0.5):
        """Detect faces.

        Arguments:
            image: a numpy uint8 array with shape [height, width, 3],
                that represents a RGB image.
            score_threshold: a float number.
        Returns:
            boxes: a float numpy array of shape [num_faces, 4].
            scores: a float numpy array of shape [num_faces].

        Note that box coordinates are in the order: ymin, xmin, ymax, xmax!
        """
        h, w, _ = image.shape
        image = np.expand_dims(image, 0)

        boxes, scores, num_boxes = self.sess.run(
            self.output_ops, feed_dict={self.input_image: image}
        )
        num_boxes = num_boxes[0]
        boxes = boxes[0][:num_boxes]
        scores = scores[0][:num_boxes]

        to_keep = scores > score_threshold
        boxes = boxes[to_keep]
        scores = scores[to_keep]

        scaler = np.array([h, w, h, w], dtype="float32")
        boxes = boxes * scaler

        return boxes, scores
