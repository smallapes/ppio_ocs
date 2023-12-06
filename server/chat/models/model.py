
import requests


def model_call(content):

    # url = "http://192.168.75.175:3002/proxy/v1/chat/completions"

    url = "http://45.116.14.16:3001/proxy/v1/chat/completions"
    # url = "http://45.116.14.16:3001/proxy/v1/chat/completions"
    # url = "https://api.openai.com/v1/completions"
    api_key = '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'

    headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json'
    }

    payload = {
    "model": "gpt-3.5-turbo",
    "messages": [
        {
        "role": "user",
        "content": content
        }
    ]
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status() # 抛出异常，如果响应码不是200
    data = response.json()
    """
    {'id': 'chatcmpl-8Sh2mhD4Q7NGnJfBxOMyuhBGT3s8G', 'object': 'chat.completion', 'created': 1701849336, 'model': 'gpt-3.5-turbo-0613', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'Hello!'}, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 9, 'completion_tokens': 2, 'total_tokens': 11}, 'system_fingerprint': None}
    """
    return data['choices'][0]["message"]["content"]


print(model_call("hello"))