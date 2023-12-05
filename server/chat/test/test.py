import unittest
import requests
import json
from server.chat.test.test_case import case1
print(case1)

def get_response(query, history):
  url = 'http://192.168.75.175:7881/chat/knowledge_base_chat'  # 这里填写你的API地址

  data = {
    "query": query,
    "knowledge_base_name": "PPIO_OPS",
    "top_k": 5,
    "score_threshold": 1,
    "history": history,
    "stream": False,
    "model_name": "chatglm-api",
    "local_doc_url": False
  }

  headers = {'Content-Type': 'application/json'}  # 根据实际情况可能需要修改

  response = requests.post(url, data=json.dumps(data), headers=headers)

  return json.loads(response.text)["answer"]


def assert_similiar(text1, text2):
  url = 'http://192.168.75.175:7881/chat/chat'  # 这里填写你的API地址

  promt = f""""
          你是一个文本相似性判断专家，可以判断两段文本是否相似。

          示例：
          文本1：你好
          文本2：您好
          回答：是
          
          文本1：我可以帮你做些什么？
          文本2：有什么可以帮助你的吗？
          回答：是

          文本1：是的，我们已经上班了
          文本2：有什么可以帮助你的吗？
          回答：否

          现在请判断下面的文本1和文本2是否大致相同，如果大致相同回答“是”，否则回答“否”：
          文本1：{text1}
          文本2：{text2}
          回答是：
          """.format(text1, text2) # 请准确判断，不能出一点差错，

  data = {
    "query": promt,
    "history": [],
    "stream": False,
    "model_name": "chatglm-api",
  }

  headers = {'Content-Type': 'application/json'}  # 根据实际情况可能需要修改

  response = requests.post(url, data=json.dumps(data), headers=headers)

  ans  = response.text
  print(65, ans)

  if  "是" in ans:
     return True
  else:
     return False


class TestBot(unittest.TestCase):
    def test_intent(self):
        history, query, ans = case1
        resp = get_response(query, history)
        self.assertEqual(resp, ans)

    def test_intent2(self):
        from server.chat.test.test_case import intent_cases
        for history, query, ans in intent_cases:
          resp = get_response(query, history)
          print(f"输出：{resp}\n参考：{ans}\n")
          llm_assert = assert_similiar(resp, ans)
          self.assertTrue(llm_assert)



if __name__ == "__main__":
   unittest.main()