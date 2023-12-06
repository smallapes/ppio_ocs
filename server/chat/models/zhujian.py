# pip install requests==2.28.1
import requests
import time


def chat_message(question = "你好，请介绍一下竹间智能"):
    # 应用ID
    appId = "2ceecc46c8f11ee95730242c0a8d0061"
    # 用户id
    user_id = "18818264891"
    # 是否是流式输出
    streaming = True
    # 问题

    # endpoint
    endpoint = "https://kkbot-llm-saas.emotibot.com"
    
    timestamp = time.time()
    # 会话sessionId
    sessionId = round(timestamp * 1000, 2)
    # 问题questionId
    questionId = round(timestamp * 1000, 2) + 1

    url = endpoint + "/chatService/chat/api/v1/chat/stream/" + user_id + "?streaming=" + str(streaming)
    data = {
        "type": "message",
        "data": {
            "question": question,
            "sessionId": sessionId,
            "appId": appId,
            "questionId": questionId
        }
    }
    headers = {
        "Content-Type": "application/json"
    }
    
    r = requests.post(url, json=data, headers=headers, stream=True)
    if r.encoding is None:
        r.encoding='utf-8'

    for lines in r.iter_lines(decode_unicode=True):
        if lines:
            print(lines)

if __name__ == "__main__":
    chat_message()