from MCP_Client.setup import *
from MCP_Client.prompt_template import *

# Helper Function
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

# Helper Function
def base64_to_pil(data_uri):
    # Strip the header "data:image/...;base64," if present
    if "," in data_uri:
        header, encoded = data_uri.split(",", 1)
    else:
        encoded = data_uri
    
    img_bytes = base64.b64decode(encoded)
    buffer = BytesIO(img_bytes)
    pil_img = Image.open(buffer)
    return pil_img

# Helper Function
def extract_base64(data_uri):
    """
    Extract the raw base64 string from a data URI like:
    'data:image/png;base64,iVBORw0KGgoAAAANSUhEUg...'
    """
    # Split once on the first comma (safe and fast)
    if "," in data_uri:
        return data_uri.split(",", 1)[1]
    else:
        # Fallback: strip via regex in case the format is unexpected
        match = re.match(r"^data:.*;base64,(.*)$", data_uri)
        if match:
            return match.group(1)
        raise ValueError("Invalid data URI: no base64 content found")

def extract_history_payload(user_query):
    with open(f"{session_path}/history.json","r") as f:
        data = json.load(f)
    history_payload = [{
                        "type":"image_url",
                        "image_url":{"url":data["image_initial_query"]}
                    },
                    {
                         "type":"text",
                         "text":"This is initial image query. \nref_number: `0`"
                    }]
    for key in data["history"].keys():
        history_payload.append({
                                    "type":"image_url",
                                    "image_url":{"url":data["history"][f"{key}"]["image_response"]}
                                })
        history_payload.append({
                                     "type":"text",
                                     "text":f"This is the #{key} request. \nThis is the resulted image based on user query \"{data['history'][key]['user_query']}\". \nref_number: `{key}`"
                                })
        
    history_payload.append({
                                "type":"text",
                                "text":f"This is the current user request. USER QUERY: \"{user_query}\""
                            })
    return history_payload

async def executing_function_calling_image_editing(user_prompt_calling, url_query_image, editing_tools, session):
    # Generate Tool Response
    response = asyncio.run(openai_client.chat.completions.create(
        model=model,
        messages=[
                {
                    "role": "system",
                    "content": system_prompt_call_editing
                },
                {
                    "role": "user", 
                    "content": [{
                                        "type":"image_url",
                                        "image_url": {"url":url_query_image}
                                },
                                {
                                        "type":"text",
                                        "text":user_prompt_calling
                                }]
                }
        ],
        tools=editing_tools,
        tool_choice="auto"))
    assistant_message = response.choices[0].message
    # Function Calling Execution
    result_messages = []
    for tool_call in assistant_message.tool_calls:
        args = json.loads(tool_call.function.arguments)
        print(f"Executing tool `{tool_call.function.name}` \nwith a payload: `{args}`")
        result = asyncio.run(session.call_tool(
                tool_call.function.name,
                arguments=args))
        result_messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result.content[0].text,
            }
        )
    url_result_image = result_messages[-1]["content"]
    # Return the output
    return url_result_image

async def get_query_image_ref_number(history_payload, routing_tools):
    # Generate Tool Response
    response = await openai_client.chat.completions.create(
        model=model,
        messages=[
                {"role": "system","content": system_prompt_call_routing},
                {"role": "user", "content": history_payload}],
        tools=routing_tools,
        tool_choice="required")
    assistant_message = response.choices[0].message
    # Function Calling Execution
    for tool_call in assistant_message.tool_calls:
        args = json.loads(tool_call.function.arguments)
        args["name"] = tool_call.function.name
        print(args)
    ref_number = args["response"]
    # Return the output
    return ref_number

async def get_reference_flagging(user_query, flagging_tools):
    # Generate Tool Response
    response = await openai_client.chat.completions.create(
        model=model,
        messages=[
                {"role": "system","content": system_prompt_call_reference},
                {"role": "user", "content": f"USER QUERY: {user_query}"}],
        tools=flagging_tools,
        tool_choice="required")
    assistant_message = response.choices[0].message
    # Function Calling Execution
    for tool_call in assistant_message.tool_calls:
        args = json.loads(tool_call.function.arguments)
        args["name"] = tool_call.function.name
        print(args)
    # Return the output
    return args["response"]

def save_chat_session(current_user_query,url_result_image):
    with open(f"{session_path}/history.json", "r+") as f:
        data = json.load(f)
        if "history" in data.keys():
            last_index = int(list(data["history"].keys())[-1])
            if (last_index+1)<=limit_history_entries:
                data["history"][last_index+1] = {"user_query":current_user_query,"image_response":url_result_image}
            else:
                appended_list = list(data["history"].values())[1:]+[{"user_query":current_user_query,"image_response":url_result_image}]
                data["history"] = {i+1:appended_list[i] for i in range(len(appended_list))}
        else:
            data["history"] = {1:{"user_query":current_user_query,"image_response":url_result_image}}
        f.seek(0)             
        json.dump(data, f, indent=4)
        f.truncate()