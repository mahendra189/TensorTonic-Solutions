import math
def generate_anchors(feature_size: int, image_size: float, scales: list[float], aspect_ratios: list[float]) -> list[list[float]]:
    """
    Returns a list of [x1, y1, x2, y2] anchor boxes.
    """
    # Write code here
    stride = image_size / feature_size

    anchors = []
    
    for i in range(feature_size):
        for j in range(feature_size):
    
            cx = (j + 0.5) * stride
            cy = (i + 0.5) * stride
    
            for scale in scales:
                for aspect_ratio in aspect_ratios:

                    # Calculate anchor dimensions
                    width = scale * math.sqrt(aspect_ratio)
                    height = scale / math.sqrt(aspect_ratio)

                    # Convert center + width/height → corners
                    x1 = cx - width / 2
                    y1 = cy - height / 2
                    x2 = cx + width / 2
                    y2 = cy + height / 2

                    anchors.append([x1, y1, x2, y2])

    return anchors