from mcp.server.fastmcp import FastMCP
from MCP_Server.setup import *
from MCP_Server.server_utils import *
from MCP_Server.prompt_template import *
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal,Optional
import re
import os
import json

class DiffusionRequest(BaseModel):
    user_query: str
    base64_img: str

class ReferenceDiffusionRequest(BaseModel):
    user_query: str
    base64_img_query: str
    base64_img_ref: str

# Create an MCP server
mcp = FastMCP(
    name="Knowledge Base",
    host="0.0.0.0",  # only used for SSE transport (localhost)
    port=8050,  # only used for SSE transport (set this to any port)
)

@mcp.resource("folder-image-session://session-asset/{folder_name}")
def show_image_list(folder_name: str) -> dict:
    """ Show List of images in the session """
    return {"path_list":os.listdir(f"session-asset/{folder_name}")}

@mcp.tool(
    name="image_reference_flagging",
    description=f"""
    {systen_image_reference_flagging}
    ================================================================================

    Input:
        response: int -> `1` if contains image reference indication and `0` otherwise | enum=[0,1]

    Output:
        response: int -> return the response
    """
)
def image_reference_flagging(response: Literal[0,1])->int:
    return response

@mcp.tool(
    name="image_query_routing",
    description=f"""
    {system_instruction_diffusion}
    ================================================================================

    Input:
        response: int -> The reference number of the query image

    Output:
        response: int -> return the response
    """
)
def image_query_routing(response: int) -> int:
    return response

@mcp.tool(
    name="editing_the_mask_without_img_reference",
    description = f"""
    {system_instruction_editing_mask_without_reference}
    ================================================================================
    
    Input:
        session_path: str -> A unified directory that may includes input of image query and image reference
        editing_prompt: str -> An editing query from the user
    
    Output:
        response: str -> base64 string of image editing result
    """
)
def editing_the_mask_without_img_reference(session_path: str, editing_prompt: str) -> str:
    list_ = os.listdir(f"{session_path}/temp")
    query_filename = [i for i in list_ if i.startswith("result")][0]
    # Convert to PIL
    query_image = Image.open(f"{session_path}/temp/{query_filename}").convert("RGB")
    ori_size = query_image.size
    # Generate image mask
    mask_image = generate_image_mask(editing_prompt, query_image, session_path)[0]
    # Generate shaded area of segmentation result
    query_image_segmented = draw_segmentation_result(query_image,mask_image)

    """ 1. Diffusion Prompt Generation """
    # Construct messages template
    messages = [
        {
            "role": "system",
            "content": system_instruction_diffusion
        },
        {
            "role": "user",
            "content": [{
                        "type": "image_url",
                        "image_url": {"url": base64_conversion(query_image)}
                        },
                        {
                        "type": "image_url",
                        "image_url": {"url": base64_conversion(query_image_segmented)}
                        },
                        {
                        "type": "text",
                        "text": f"User: {editing_prompt}"
                         }]  
        }
    ]
    # Llama4 inference
    diffusion_prompt = llama4_inference(messages)
    # Extract informaton
    diffusion_prompt_payload = json.loads(re.search(r'{.*}', diffusion_prompt, re.DOTALL).group(0))
    print(diffusion_prompt_payload)

    """ 2. Diffusion Model Inference """
    # Diffusion pipeline
    result_image = prefpaint_pipe(
      prompt=diffusion_prompt_payload["positive_prompt"],
      negative_prompt= diffusion_prompt_payload["negative_prompt"],
      image=query_image,
      mask_image=clean_mask(mask_image, kernel_size=5),
      eta=1.0,
      padding_mask_crop=5,
      generator=torch.Generator(device=device_1).manual_seed(0)
    ).images[0].resize(ori_size)
    # Save the result
    result_image_saved = result_image.copy()
    result_image_saved.save(f"{session_path}/temp/{query_filename}") 
    # Convert to base64 and return the output
    return base64_conversion(result_image)

@mcp.tool(
    name="editing_the_mask_with_img_reference",
    description = f"""
    {system_instruction_editing_the_mask_with_img_reference}
    ================================================================================

    Input:
        session_path: str -> A unified directory that may includes input of image query and image reference
        editing_prompt: str -> An editing query from the user
    
    Output:
        response: str -> base64 string of image editing result
    """
)
def editing_the_mask_with_img_reference(session_path: str, editing_prompt: str) -> str:
    list_ = os.listdir(f"{session_path}/temp")
    query_filename = [i for i in list_ if i.startswith("result")][0]
    ref_filename = [i for i in list_ if i.startswith("ref")][0]
    # Convert to PIL
    query_image = Image.open(f"{session_path}/temp/{query_filename}").convert("RGB")
    ori_size = query_image.size
    # Generate image mask
    mask_image = generate_image_mask(editing_prompt, query_image, session_path)[0]
    # Open Reference image path
    reference_image = Image.open(f"{session_path}/temp/{ref_filename}").convert("RGB")
    # Diffusion model inference
    result_image = ip_model.generate(
        pil_image=reference_image,
        num_samples=1,
        num_inference_steps=50,
        seed=0,
        image=query_image,
        mask_image=clean_mask(mask_image, kernel_size=5),
        padding_mask_crop=5,
        guidance_scale=6.5,
        strength=0.99)[0].resize(ori_size)   
    # Save the result
    result_image_saved = result_image.copy()
    result_image_saved.save(f"{session_path}/temp/{query_filename}") 
    # Convert to base64 and return the output
    return base64_conversion(result_image)

@mcp.tool(
    name="object_insertion",
    description = f"""
    {system_instruction_add_object}
    ================================================================================

    Input:
        session_path: str -> A unified directory that includes input of image query and image reference
        prompt_source: str -> Source prompt for generating the base image
        prompt_target: str -> Target prompt describing the desired edited image
        subject_token: str -> Single token representing the subject to add, this token must appear in `prompt_target`

    Output:
        response: str -> base64 string of edited image result
    """
)
def object_insertion(session_path: str, prompt_source: str, prompt_target: str, subject_token:str) -> str:
    list_ = os.listdir(f"{session_path}/temp")
    query_filename = [i for i in list_ if i.startswith("result")][0]
    # Reset the GPU memory tracking
    torch.cuda.reset_max_memory_allocated(0)
    query_image = Image.open(f"{session_path}/temp/{query_filename}").convert("RGB")
    ori_size = query_image.size
    random.seed(0)
    seed_src = random.randint(0, 10000)
    seeds_obj = random.randint(0, 3)
    source_img, result_img = add_object_real(addit_pipe, source_image=query_image, prompt_source=prompt_source, prompt_object=prompt_target, 
                            subject_token=subject_token, seed_src=seed_src, seed_obj=seeds_obj, 
                            extended_scale =1.1, structure_transfer_step=4, blend_steps = [18], #localization_model="attention",
                            use_offset=False, show_attention=True, use_inversion=True, display_output=True)
    result_img = result_img.resize(ori_size)
    # Save the result
    source_img.save(f"{session_path}/temp/source.png") 
    # Save the result
    result_img.save(f"{session_path}/temp/{query_filename}") 
    print(base64_conversion(result_img))
    # Convert to base64 and return the output
    return base64_conversion(result_img)

@mcp.tool(
    name="fashion_try_on_without_img_reference",
    description = f"""
    {system_instruction_fashion_try_on_without_img_reference}
    ================================================================================

    Input:
        session_path: str -> A unified directory that may includes input of image query and image reference
        editing_prompt: str -> An editing query from the user
        diffusion_prompt: str -> A clear description of the cloth/garment to be generated by diffusion model
    
    Output:
        response: str -> base64 string of image editing result
    """
)
def fashion_try_on_without_img_reference(session_path: str, editing_prompt: str, diffusion_prompt: str) -> str:
    list_ = os.listdir(f"{session_path}/temp")
    query_filename = [i for i in list_ if i.startswith("result")][0]
    query_image = Image.open(f"{session_path}/temp/{query_filename}").convert("RGB")
    ori_size = query_image.size
    # Image Segmentation
    mask_image = generate_image_mask(editing_prompt, query_image, session_path)[0]
    # Diffusion Model Inference (generate garment)
    cloth_image = sdxl_pipe(prompt=diffusion_prompt).images[0]
    cloth_image.save(f"{session_path}/temp/clolth_ref.png") 
    cloth_image = resize_and_padding(cloth_image, query_image.size)
    # Inpainting
    result_image = vton_pipeline(
        image=query_image,
        condition_image=cloth_image,
        mask=clean_mask(mask_image, kernel_size=5),
        num_inference_steps=50,
        guidance_scale=2.5,
        generator=None
    )[0]
    # Save the result
    result_image.save(f"{session_path}/temp/{query_filename}") 
    # Convert to base64 and return the output
    return base64_conversion(result_image)

@mcp.tool(
    name="fashion_try_on_with_img_reference",
    description = f"""
    {system_instruction_fashion_try_on_with_img_reference}
    ================================================================================

    Input:
        session_path: str -> A unified directory that may includes input of image query and image reference
        editing_prompt: str -> An editing query from the user
    
    Output:
        response: str -> base64 string of image editing result
    """
)
def fashion_try_on_with_img_reference(session_path: str, editing_prompt: str) -> str:
    list_ = os.listdir(f"{session_path}/temp")
    query_filename = [i for i in list_ if i.startswith("result")][0]
    ref_filename = [i for i in list_ if i.startswith("ref")][0]
    query_image = Image.open(f"{session_path}/temp/{query_filename}").convert("RGB")
    ori_size = query_image.size
    cloth_image = Image.open(f"{session_path}/temp/{ref_filename}").convert("RGB")
    # Image Segmentation
    mask_image = generate_image_mask(editing_prompt, query_image, session_path)[0]
    # Resize and padding
    cloth_image = resize_and_padding(cloth_image, query_image.size)
    # Inpainting
    result_image = vton_pipeline(
        image=query_image,
        condition_image=cloth_image,
        mask=clean_mask(mask_image, kernel_size=5),
        num_inference_steps=50,
        guidance_scale=2.5,
        generator=None
    )[0]
    # Save the result
    result_image.save(f"{session_path}/temp/{query_filename}") 
    # Convert to base64 and return the output
    return base64_conversion(result_image)

@mcp.tool(
    name="object_removal",
    description = f"""
    {system_instruction_object_removal}
    ================================================================================

    Input:
        session_path: str -> A unified directory that may includes input of image query and image reference
        editing_prompt: str -> An editing query from the user
    
    Output:
        response: str -> base64 string of image editing result
    """
)
def object_removal(session_path: str, editing_prompt: str) -> str:
    list_ = os.listdir(f"{session_path}/temp")
    query_filename = [i for i in list_ if i.startswith("result")][0]
    # Convert to PIL
    query_image = Image.open(f"{session_path}/temp/{query_filename}").convert("RGB")
    ori_size = query_image.size
    query_image = resize_by_short_side(query_image, 512, resample=Image.BICUBIC)
    # Generate image mask
    mask_image = generate_image_mask(editing_prompt, query_image, session_path)[0]
    """ Diffusion Model Inference """
    # Diffusion pipeline
    w, h = query_image.size
    result_image = removal_pipe(
            prompt="remove the instance of object",
            image=query_image,
            mask_image=clean_mask(mask_image),
            generator=generator,
            num_inference_steps=20,
            guidance_scale=2.5,
            height=h,
            width=w,
            return_attn_map=False,
        ).images[0].resize(ori_size)
    # Save the result
    result_image_saved = result_image.copy()
    result_image_saved.save(f"{session_path}/temp/{query_filename}") 
    # Convert to base64 and return the output
    return base64_conversion(result_image)

@mcp.tool(
    name="text_editing",
    description = f"""
    {system_instruction_text_editing}
    ================================================================================

    Input:
        session_path: str -> A unified directory that may includes input of image query and image reference
        editing_prompt: str -> An editing query from the user
    
    Output:
        response: str -> base64 string of image editing result
    """
)
def text_editing(session_path: str, editing_prompt: str, text: str, color: str) -> str:
    list_ = os.listdir(f"{session_path}/temp")
    query_filename = [i for i in list_ if i.startswith("result")][0]
    # Convert to PIL
    query_image = Image.open(f"{session_path}/temp/{query_filename}").convert("RGB")
    ori_size = query_image.size
    query_image = resize_by_short_side(query_image, 512, resample=Image.BICUBIC)
    # Generate image mask
    mask_image, bboxes, points = generate_image_mask(editing_prompt, query_image, session_path)
    """ Diffusion Model Inference """
    # Diffusion pipeline
    w, h = query_image.size
    result_image = removal_pipe(
            prompt="remove the text",
            image=query_image,
            mask_image=clean_mask(mask_image, kernel_size=5),
            generator=generator,
            num_inference_steps=20,
            guidance_scale=2.5,
            height=h,
            width=w,
            return_attn_map=False,
        ).images[0]
    """ Draw the Text """
    result_image = draw_text_in_boxes(result_image, bboxes, [text], color).resize(ori_size)
    # Save the result
    result_image_saved = result_image.copy()
    result_image_saved.save(f"{session_path}/temp/{query_filename}") 
    # Convert to base64 and return the output
    return base64_conversion(result_image)

@mcp.tool(
    name="object_localization",
    description = f"""
    {system_instruction_object_localization}
    ================================================================================

    Input:
        session_path: str -> A unified directory that may includes input of image query and image reference
        editing_prompt: str -> An editing query from the user
    
    Output:
        response: str -> base64 string of image editing result
    """
)
def object_localization(session_path: str, editing_prompt: str) -> str:
    list_ = os.listdir(f"{session_path}/temp")
    query_filename = [i for i in list_ if i.startswith("result")][0]
    # Convert to PIL
    query_image = Image.open(f"{session_path}/temp/{query_filename}").convert("RGB")
    # Generate bounding boxes
    bboxes = generate_image_mask(editing_prompt, query_image, session_path)[1]
    """ Draw the Bounding Boxes """
    result_image = draw_bboxes(query_image, bboxes)
    # Save the result
    result_image_saved = result_image.copy()
    result_image_saved.save(f"{session_path}/temp/{query_filename}") 
    # Convert to base64 and return the output
    return base64_conversion(result_image)

# Run the server
if __name__ == "__main__":
    mcp.run(transport="sse")
