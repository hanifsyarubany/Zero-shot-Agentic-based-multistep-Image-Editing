from MCP_Server.setup import *
from MCP_Server.prompt_template import *

def llama4_inference(messages, token=1024):
    completion = client_groq.chat.completions.create(
        model="meta-llama/llama-4-maverick-17b-128e-instruct",
        messages=messages,
        temperature=0.1,
        max_completion_tokens=token,
        top_p=1,
        stream=True,
        stop=None,
    )
    inference_result = ""
    for chunk in completion:
        chunk_inference = chunk.choices[0].delta.content or ""
        inference_result += chunk_inference
    text = inference_result
    return text

def gpt_completion(messages):
    response = client_gpt.responses.create(
                model="gpt-5-nano",
                input=messages,
                text={
                    "format": {
                    "type": "text"
                    },
                    "verbosity": "medium"
                },
                reasoning={
                    "effort": "medium"
                },
                tools=[],
                store=True)
    return response

def base64_conversion(pil_img):
    buffer = BytesIO()
    # If format is unknown, use PNG to avoid JPEG compression artifacts
    format = pil_img.format or "PNG"
    # Optional: force convert to RGB to avoid issues with transparency in JPEG
    if format.upper() == "JPEG" and pil_img.mode in ("RGBA", "P"):
        pil_img = pil_img.convert("RGB")
    pil_img.save(buffer, format=format, quality=95)  # high quality for JPEG
    img_bytes = buffer.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/{format.lower()};base64,{img_base64}"

def raw_base64_conversion(pil_img):
    buffer = BytesIO()
    # If format is unknown, use PNG to avoid JPEG compression artifacts
    format = pil_img.format or "PNG"
    # Optional: force convert to RGB to avoid issues with transparency in JPEG
    if format.upper() == "JPEG" and pil_img.mode in ("RGBA", "P"):
        pil_img = pil_img.convert("RGB")
    pil_img.save(buffer, format=format, quality=95)  # high quality for JPEG
    img_bytes = buffer.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    return img_base64

def pil_converstion(base64_img):
    # Decode the base64 string
    image_data = base64.b64decode(base64_img)
    # Wrap the binary data with BytesIO
    image_io = BytesIO(image_data)
    # Open the image using PIL
    image = Image.open(image_io).convert("RGB")
    # Output
    return image

def draw_segmentation_result(
    image: Image.Image,
    mask_pil: Image.Image,
    highlight_rgba=(255, 0, 0, 128),
    darken_factor: float = 0.5,
    show: bool = True,
):
    """
    image: PIL.Image (any mode) - base image
    mask_pil: PIL.Image (any mode) - nonzero/white pixels are treated as mask
    highlight_rgba: RGBA tuple for the mask overlay
    darken_factor: 0..1 amount to darken non-mask area
    show: whether to .show() the result; function returns the blended image either way
    """

    # Ensure base image is RGBA
    img = image.convert("RGBA")

    # Make sure mask is same size; convert to single-channel L and binarize
    if mask_pil.size != img.size:
        mask_pil = mask_pil.resize(img.size, Image.NEAREST)
    mask_L = mask_pil.convert("L")
    mask = np.array(mask_L) > 0  # boolean mask where nonzero = True

    # Convert base image to ndarray
    img_np = np.array(img, dtype=np.uint8)

    # --- Create red highlight overlay on masked area ---
    overlay_highlight = np.zeros_like(img_np, dtype=np.uint8)
    r, g, b, a = highlight_rgba
    overlay_highlight[mask] = np.array([r, g, b, a], dtype=np.uint8)

    # --- Create darkened base on non-mask area (RGB only) ---
    overlay_darken = img_np.copy()
    non_mask = ~mask
    # Multiply only RGB channels for non-mask pixels
    overlay_darken[non_mask, :3] = (overlay_darken[non_mask, :3].astype(np.float32) * darken_factor).clip(0, 255).astype(np.uint8)
    # Keep alpha fully opaque for the base
    overlay_darken[..., 3] = 255

    # --- Composite: darkened base + red highlight on top ---
    blended = Image.alpha_composite(Image.fromarray(overlay_darken, mode="RGBA"),
                                    Image.fromarray(overlay_highlight, mode="RGBA"))
    return blended

def generate_image_mask(user_query, query_image, session_path):
    """ 1. Object Localization """
    # Construct messages
    messages = [
        {
            "role": "system",
            "content": system_instruction_segmentation
        },
        {
            "role": "user",
            "content": [{
                        "type": "text",
                        "text": f"User: {user_query}"
                         },
                        {
                        "type": "image_url",
                        "image_url": {"url": base64_conversion(query_image)}
                        }]  
        }
    ]
    # Creating segmentation prompt
    segmentation_prompt = llama4_inference(messages)
    print(f"SEGMENTATION PROMPT = {segmentation_prompt}")
    # segmentation_prompt = user_query
    # print(segmentation_prompt)
    # Prepare image (Image Resizing)
    original_width, original_height = query_image.size
    x_factor, y_factor = original_width/resize_size, original_height/resize_size
    resized_query_image = query_image.resize((resize_size, resize_size), Image.BILINEAR)
    # Format text based on template
    formatted_text = DETECTION_TEMPLATE.format(
        Instruction=segmentation_prompt,
        Answer="[{\"bbox_2d\": [10,100,200,210], \"point_2d\": [30,110]}, {\"bbox_2d\": [225,296,706,786], \"point_2d\": [302,410]}]"
    )
    # Construct messages
    message = [{
                "role": "user",
                "content": [
                    {
                        "type": "image", 
                        "image": resized_query_image
                    },
                    {   
                        "type": "text",
                        "text": formatted_text
                    }
                ]
            }]
    # VisionReasoner inference
    inputs = vision_reasoner_processor(
            text=vision_reasoner_processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True),
            images=resized_query_image,
            padding=True,
            return_tensors="pt",
        ).to(device_1)
    generated_ids = vision_reasoner.generate(**inputs, use_cache=True, max_new_tokens=2048, do_sample=False)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = vision_reasoner_processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False)[0]
    # Print out the thinking
    print(output_text)
    # Data Extraction
    json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', output_text, re.DOTALL)
    pred_bboxes = []
    pred_points = []
    pred_answer = None
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            pred_answer = data
            pred_bboxes = [[
                int(item['bbox_2d'][0] * x_factor + 0.5),
                int(item['bbox_2d'][1] * y_factor + 0.5),
                int(item['bbox_2d'][2] * x_factor + 0.5),
                int(item['bbox_2d'][3] * y_factor + 0.5)
            ] for item in data]
            pred_points = [[
                int(item['point_2d'][0] * x_factor + 0.5),
                int(item['point_2d'][1] * y_factor + 0.5)
            ] for item in data]
        except Exception as e:
            print(f"Error parsing JSON: {e}")
    bboxes, points = pred_bboxes, pred_points

    """ 2. Image Segmentation """
    segmentation_model.set_image(query_image)
    img_height, img_width = query_image.height, query_image.width
    mask_arr = np.zeros((img_height, img_width), dtype=bool)
    for bbox, point in zip(bboxes, points):
        masks, scores, _ = segmentation_model.predict(
            point_coords=[point],
            point_labels=[1],
            box=bbox)
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        mask = masks[0].astype(bool)
        mask_arr = np.logical_or(mask_arr, mask)
    mask_image = Image.fromarray(mask_arr).convert('L')
    draw_segmentation_result(query_image, mask_image).save(f"{session_path}/temp/mask_image.png", "PNG")
    # Return Output
    return mask_image, bboxes, points

def clean_mask(pil_mask, kernel_size=15):
    # Convert to numpy boolean mask
    arr = np.array(pil_mask.convert("L")) > 127
    # Morphological closing to fill small black holes
    cleaned = binary_closing(arr, structure=np.ones((kernel_size, kernel_size)))
    # Back to PIL Image
    return Image.fromarray((cleaned * 255).astype(np.uint8))

def draw_text_in_boxes(image, boxes, texts, color, font_path=None, font_size=1000):
    image_ = image.copy()
    draw = ImageDraw.Draw(image_)
    def get_text_size(txt, fnt):
        """Cross-version text size getter (Pillow <10 vs >=10)."""
        try:
            bbox = draw.textbbox((0, 0), txt, font=fnt)
            return bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            return draw.textsize(txt, font=fnt)
    for box, text in zip(boxes, texts):
        x1, y1, x2, y2 = box
        box_w, box_h = x2 - x1, y2 - y1
        # Start with given font size
        current_size = font_size
        while current_size > 1:
            try:
                font = ImageFont.truetype(font_path or "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", current_size)
            except OSError:
                font = ImageFont.load_default()
            text_w, text_h = get_text_size(text, font)
            if text_w <= box_w and text_h <= box_h:
                break  # Fits inside the box
            current_size -= 1
        # Center the text
        text_x = x1 + (box_w - text_w) / 2
        text_y = y1 + (box_h - text_h) / 2
        draw.text((text_x, text_y), text, fill=color, font=font)
    return image_

def draw_bboxes(pil_image, bboxes):
    image = pil_image.copy()
    draw = ImageDraw.Draw(image)
    # Draw bounding boxes (blue rectangles)
    if bboxes:
        for box in bboxes:
            x0, y0, x1, y1 = box
            draw.rectangle([x0, y0, x1, y1], outline="red", width=6)
    return image