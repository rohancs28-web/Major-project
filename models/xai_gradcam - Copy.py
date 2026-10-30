import tensorflow as tf
import numpy as np
import cv2

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) for CNN Spatial Feature Interpretation.
    """
    def __init__(self, model, last_conv_layer_name):
        self.model = model
        self.last_conv_layer_name = last_conv_layer_name
        
        # Create a sub-model that maps input to the last conv layer and the final output
        self.grad_model = tf.keras.models.Model(
            inputs=self.model.inputs,
            outputs=[
                self.model.get_layer(self.last_conv_layer_name).output,
                self.model.output
            ]
        )

    def generate_heatmap(self, image_tensor):
        """
        Generates a Grad-CAM heatmap for a given input image tensor.
        """
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(image_tensor)
            loss = tf.reduce_mean(predictions)

        # Gradient of the output with respect to the output feature map of the last conv layer
        grads = tape.gradient(loss, conv_outputs)
        
        # Vector of mean intensity of gradients over specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize heatmap between 0 and 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()

    def overlay_heatmap(self, original_image, heatmap, alpha=0.4):
        """
        Overlays the generated heatmap onto the original satellite image patch.
        """
        heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        superimposed_img = heatmap * alpha + original_image * (1 - alpha)
        return np.uint8(superimposed_img)