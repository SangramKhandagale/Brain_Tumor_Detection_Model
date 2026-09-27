"""
gradcam.py
----------
Grad-CAM explainability for the brain tumor CNN.

Given a trained tf.keras model and a preprocessed input image, this module:
  1. Finds which pixels most influenced the model's decision (Grad-CAM heatmap).
  2. Overlays that heatmap on the original MRI for visual display.
  3. Extracts discrete "hotspot" regions (bounding boxes + centroid + size)
     from the heatmap so a report generator can describe *where* in the
     brain the model focused, in plain-language terms.

No layer names are hardcoded — it automatically finds the last Conv2D layer,
so this works whether the model is the current Sequential CNN or a deeper
one (ResNet/EfficientNet-style) later on.
"""

import numpy as np
import tensorflow as tf
import cv2


def get_last_conv_layer_name(model: tf.keras.Model) -> str:
    """Return the name of the last Conv2D (or DepthwiseConv2D/SeparableConv2D)
    layer in the model. Grad-CAM needs a conv layer's feature maps."""
    for layer in reversed(model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D,
                               tf.keras.layers.SeparableConv2D,
                               tf.keras.layers.DepthwiseConv2D)):
            return layer.name
    raise ValueError(
        "No Conv2D layer found in the model — Grad-CAM requires at least one "
        "convolutional layer to read feature maps from."
    )


def make_gradcam_heatmap(img_array: np.ndarray,
                          model: tf.keras.Model,
                          last_conv_layer_name: str = None,
                          pred_index: int = None):
    """
    Compute a Grad-CAM heatmap.

    img_array: preprocessed batch of shape (1, H, W, C), same as what you
               feed to model.predict().
    pred_index: which class's score to explain. Defaults to the model's
                top predicted class, so the heatmap always explains
                "why did you pick THIS class".

    Returns: (heatmap, pred_index, class_confidence)
             heatmap is a float32 array in [0, 1], shape (h, w) at the
             conv layer's spatial resolution (later resized to the image).
    """
    if last_conv_layer_name is None:
        last_conv_layer_name = get_last_conv_layer_name(model)

    # NOTE ON IMPLEMENTATION: the "standard" Grad-CAM recipe builds a
    # sub-model via `Model(inputs=model.inputs,
    # outputs=[model.get_layer(name).output, model.output])`. That works
    # for functional models, but Keras 3's Sequential models have a bug
    # where the resulting sub-model's gradient w.r.t. the intermediate
    # layer comes back as None — the two outputs end up disconnected in
    # the tape even though the forward values are correct. Since this
    # project's model is Sequential, we instead replay each layer's
    # __call__ manually inside the tape, which sidesteps the bug and
    # works for Sequential and functional models alike.
    img_tensor = tf.convert_to_tensor(img_array)
    conv_outputs = None
    x = img_tensor
    with tf.GradientTape() as tape:
        for layer in model.layers:
            x = layer(x, training=False)
            if layer.name == last_conv_layer_name:
                tape.watch(x)
                conv_outputs = x
        predictions = x
        if conv_outputs is None:
            raise ValueError(
                f"Layer '{last_conv_layer_name}' was not found while "
                f"replaying the model's layers — check the layer name."
            )
        if pred_index is None:
            pred_index = int(tf.argmax(predictions[0]))
        class_channel = predictions[:, pred_index]

    # Gradient of the predicted class score w.r.t. the conv feature maps
    grads = tape.gradient(class_channel, conv_outputs)

    # Global-average-pool the gradients -> importance weight per channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU: we only care about features that *positively* influenced the class
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    confidence = float(predictions[0][pred_index])
    return heatmap.numpy(), pred_index, confidence


def overlay_heatmap(original_image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45):
    """
    Resize heatmap to the original image size and blend it as a color
    overlay (jet colormap) on top of the original MRI. Returns an RGB
    uint8 image ready for st.image().
    """
    if original_image.dtype != np.uint8:
        original_image = np.clip(original_image, 0, 255).astype(np.uint8)

    h, w = original_image.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

    if original_image.ndim == 2:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)

    overlaid = cv2.addWeighted(original_image, 1 - alpha, colored_heatmap, alpha, 0)
    return overlaid, heatmap_resized


def get_hotspot_regions(heatmap_resized: np.ndarray, threshold: float = 0.5, min_area_frac: float = 0.005):
    """
    Turn a continuous heatmap into discrete "hotspot" regions a report can
    reference: bounding box, centroid (normalized 0-1), area fraction of
    the whole image, and peak activation strength.

    threshold: fraction of max activation above which a pixel counts as
               part of a hotspot (0.5 = top half of the activation range).
    min_area_frac: ignore tiny noise blobs smaller than this fraction of
                   total image area.

    Returns a list of dicts, sorted by peak activation (strongest first):
        {
          "bbox": (x, y, w, h),          # pixel coords in the resized image
          "centroid_norm": (cx, cy),     # 0-1 normalized position
          "area_frac": float,            # fraction of image area
          "peak_activation": float,      # 0-1
          "mean_activation": float,      # 0-1, average strength in the region
        }
    """
    h, w = heatmap_resized.shape
    total_area = h * w

    binary_mask = (heatmap_resized >= threshold).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    regions = []
    for label_id in range(1, num_labels):  # skip background label 0
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area / total_area < min_area_frac:
            continue

        x = stats[label_id, cv2.CC_STAT_LEFT]
        y = stats[label_id, cv2.CC_STAT_TOP]
        bw = stats[label_id, cv2.CC_STAT_WIDTH]
        bh = stats[label_id, cv2.CC_STAT_HEIGHT]
        cx, cy = centroids[label_id]

        region_mask = labels == label_id
        peak = float(heatmap_resized[region_mask].max())
        mean = float(heatmap_resized[region_mask].mean())

        regions.append({
            "bbox": (int(x), int(y), int(bw), int(bh)),
            "centroid_norm": (cx / w, cy / h),
            "area_frac": area / total_area,
            "peak_activation": peak,
            "mean_activation": mean,
        })

    regions.sort(key=lambda r: r["peak_activation"], reverse=True)
    return regions


def draw_hotspot_boxes(image: np.ndarray, regions: list, color=(0, 255, 100), thickness=2):
    """Draw bounding boxes + rank labels for each hotspot on a copy of the image."""
    annotated = image.copy()
    for i, region in enumerate(regions):
        x, y, w, h = region["bbox"]
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)
        cv2.putText(annotated, f"#{i+1}", (x, max(y - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return annotated