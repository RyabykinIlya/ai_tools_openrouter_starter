import logging
import os
import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOGLEVEL", "ERROR"))
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s: %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def request_openrouter_llm(data, api_key, url):
    return requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        json=data
    )


hints = [
    "Будь вежлив",
    "Не причиняй вред человеку",
    "Говори правду и признавай свои ограничения",
    "Уважай приватность пользователя",
    "Действуй прозрачно и объясняй свои решения",
    "Сохраняй беспристрастность и избегай предвзятости",
    "Стремись к пользе для человечества"
]

def get_next_hint():
    get_next_hint.current = getattr(get_next_hint, 'current', -1) + 1
    if get_next_hint.current >= len(hints):
        get_next_hint.current = 0
    return hints[get_next_hint.current]


def handle_tool_calls(message, messages):
    if "tool_calls" in message:
        for tool_call in message["tool_calls"]:
            logger.info("Вызываю функцию %s" % tool_call["function"]["name"])
            if tool_call["function"]["name"] == "get_next_hint":
                hint = get_next_hint()
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": "get_next_hint",
                    "content": hint
                })
            # elif tool_call["function"]["name"] == "execute windows command":
            else:
                raise NotImplementedError(f"Tool {tool_call['function']['name']} not implemented")

def askOpenRouter(
    prompt: str,
    model_name:str,
    temperature:int,
    api_key:str,
    messages: list,
    tools: list,
):
    """
    Send request to OpenRouter API
    Documentation: https://openrouter.ai/docs
    """
    
    data = {
        "model": model_name,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": prompt},
            *messages,
        ],
        "tools": tools,
        "tool_choice": "auto"
    }

    logger.debug("OpenRouter request data: %s" % data)
    response = request_openrouter_llm(data, api_key, "https://openrouter.ai/api/v1/chat/completions").json()
    logger.debug("OpenRouter response: %s" % response)
    
    message = response["choices"][0]["message"]
    if "tool_calls" in message:
        handle_tool_calls(message, messages)
        # Make another request to get the response after tool execution
        return askOpenRouter(prompt, model_name, temperature, api_key, messages, tools)
    
    return message["content"]

# In the main loop, modify the response handling:
if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    # модель можно выбрать тут: https://openrouter.ai/models, сортируй от самых дешёвых 
    model_name = "google/gemini-2.0-pro-exp-02-05:free"
    
    # Initialize chat history
    messages = []
    system_prompt = "You are a helpful AI assistant."

    # про тулз коллинг https://openrouter.ai/docs/features/tool-calling
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_next_hint",
                "description": "Получить заповедь, если пользователь просит",
            }
        }
    ]
    
    print("Chat with AI (type 'quit' or press Ctrl+C to exit)")
    first_message = True
    while True:
        if first_message:
            print("\nПодсказка: Спроси заповедь")
            first_message = False
            
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
            
        messages.append({"role": "user", "content": user_input})
        
        ai_response = askOpenRouter(
            prompt=system_prompt,
            model_name=model_name,
            temperature=1.3,
            api_key=api_key,
            messages=messages,
            tools=tools
        )
        
        print("AI:", ai_response)
        messages.append({"role": "assistant", "content": ai_response})

