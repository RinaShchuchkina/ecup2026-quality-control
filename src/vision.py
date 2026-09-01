import math

from PIL import Image

MAX_RATIO = 200
SPATIAL_MERGE_SIZE = 2
IMAGE_MIN_TOKEN_NUM = 4
IMAGE_MAX_TOKEN_NUM = 16384
DEFAULT_PATCH_SIZE = 14


def round_by_factor(number, factor):
    return round(number / factor) * factor


def ceil_by_factor(number, factor):
    return math.ceil(number / factor) * factor


def floor_by_factor(number, factor):
    return math.floor(number / factor) * factor


def smart_resize(height, width, factor, min_pixels=None, max_pixels=None):
    max_pixels = max_pixels if max_pixels is not None else (IMAGE_MAX_TOKEN_NUM * factor ** 2)
    min_pixels = min_pixels if min_pixels is not None else (IMAGE_MIN_TOKEN_NUM * factor ** 2)
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError("aspect ratio too extreme")
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def to_rgb(pil_image):
    if pil_image.mode == "RGBA":
        white = Image.new("RGB", pil_image.size, (255, 255, 255))
        white.paste(pil_image, mask=pil_image.split()[3])
        return white
    return pil_image.convert("RGB")


def fetch_image(ele, image_patch_size=DEFAULT_PATCH_SIZE):
    patch_factor = int(image_patch_size * SPATIAL_MERGE_SIZE)
    image = to_rgb(Image.open(ele["image"]))
    pre = ele.get("pre_max_side")
    if pre:
        from io import BytesIO
        img2 = image.copy()
        if max(img2.size) > pre:
            img2.thumbnail((pre, pre))
        buf = BytesIO()
        img2.save(buf, "JPEG", quality=87)
        buf.seek(0)
        image = Image.open(buf).convert("RGB")
    width, height = image.size
    resized_height, resized_width = smart_resize(
        height, width, factor=patch_factor,
        min_pixels=ele.get("min_pixels"), max_pixels=ele.get("max_pixels"),
    )
    return image.resize((resized_width, resized_height))


def collect_images(conversations):
    images = []
    for conv in conversations:
        for msg in conv:
            content = msg.get("content")
            if isinstance(content, list):
                for ele in content:
                    if isinstance(ele, dict) and ele.get("type") == "image":
                        images.append(fetch_image(ele))
    return images or None
