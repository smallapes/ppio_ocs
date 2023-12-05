from fastapi import Body, Request
from fastapi.responses import StreamingResponse
from configs.model_config import (llm_model_dict, LLM_MODEL, PROMPT_TEMPLATE,
                                  VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD)
from server.chat.utils import wrap_done
from server.utils import BaseResponse
from langchain.chat_models import ChatOpenAI
from langchain.llms import  OpenAI
from langchain import LLMChain
from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import AsyncIterable, List, Optional
import asyncio
from langchain.prompts.chat import ChatPromptTemplate
from server.chat.utils import History
from server.knowledge_base.kb_service.base import KBService, KBServiceFactory
import json
import os
from urllib.parse import urlencode
from server.knowledge_base.kb_doc_api import search_docs
from server.chat.prompts import intent_prompt, instructions, instructions_q, guidance_prompt, analyze_prompt, qa_prompt, call_artificial, inquiry_prompt, functions, device_info_template
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
from server.chat.utils import extract_id
from server.chat.tools.tools_api import DeviceLoad, DeviceNATInfo, DeviceTestInfo, DeviceUsageInfo, TaskRedLine, ChangePointInfo, DeviceTaskid
import re

def knowledge_base_chat(query: str = Body(..., description="用户输入", examples=["你好"]),
                        knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
                        top_k: int = Body(VECTOR_SEARCH_TOP_K, description="匹配向量数"),
                        score_threshold: float = Body(SCORE_THRESHOLD, description="知识库匹配相关度阈值，取值范围在0-1之间，SCORE越小，相关度越高，取到1相当于不筛选，建议设置在0.5左右", ge=0, le=1),
                        history: List[History] = Body([],
                                                      description="历史对话",
                                                      examples=[[
                                                          {"role": "user",
                                                           "content": "我们来玩成语接龙，我先来，生龙活虎"},
                                                          {"role": "assistant",
                                                           "content": "虎头虎脑"}]]
                                                      ),
                        stream: bool = Body(False, description="流式输出"),
                        model_name: str = Body(LLM_MODEL, description="LLM 模型名称。"),
                        local_doc_url: bool = Body(False, description="知识文件返回本地路径(true)或URL(false)"),
                        request: Request = None,
                        ):
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    history = [History.from_data(h) for h in history]
    docs = search_docs(query, knowledge_base_name, top_k, score_threshold)

    # 意图分析
    from enum import Enum
    class Intent(Enum):
        qa = "智能问答"
        guidance = "智能引导"
        analyze = "跑量诊断"
        artificial ="转人工"
        unclear = "不清楚"

    def get_intent(query: str,
                kb: KBService,
                top_k: int,
                history: Optional[List[History]],
                model_name: str = LLM_MODEL,
                ):
        qs = "\n".join([doc.page_content.split("答案")[0] for doc in docs])
        
        # prompt = PromptTemplate.from_template(intent_prompt)

        prompt = intent_prompt.format(qs=qs, instructions_q=instructions_q, user_input=query)
        model = OpenAI(
            streaming=True,
            verbose=True,
            # callbacks=[callback],
            openai_api_key=llm_model_dict[model_name]["api_key"],
            openai_api_base=llm_model_dict[model_name]["api_base_url"],
            model_name=model_name,
            openai_proxy=llm_model_dict[model_name].get("openai_proxy"),
            temperature = 0,
            top_p = 0.5
        )

        input_msg = History(role="user", content=prompt).to_msg_template()
        chat_prompt = ChatPromptTemplate.from_messages(
            [i.to_msg_template() for i in history] + [input_msg])
        
        prompt = chat_prompt.format()

        response = model.invoke(prompt)

        def get_intent(text):
            # 定义关键词列表
            keywords = ['智能问答', '跑量诊断', '智能引导', '不清楚']

            # 初始化计数器字典
            count_dict = {keyword: 0 for keyword in keywords}

            # 遍历关键词，使用正则表达式进行搜索并统计出现次数
            for keyword in keywords:
                pattern = re.compile(keyword)
                matches = pattern.findall(text)
                count_dict[keyword] = len(matches)
            intent_list = sorted(count_dict, key=lambda x: count_dict[x], reverse=True)
            intent = intent_list[0]
            if count_dict[intent] == 0:
                intent = "转人工"
            if "智能问答" in intent:
                return Intent.qa
            elif "智能引导" in intent:
                return Intent.guidance
            elif "跑量诊断" in intent:
                return Intent.analyze
            elif "不清楚" in intent:
                return Intent.unclear
            else:
                return Intent.artificial

        return get_intent(response)

        
    intent = get_intent(query, kb, top_k, history, model_name)


    # 智能问答
    async def knowledge_base_chat_iterator(query: str,
                                           kb: KBService,
                                           top_k: int,
                                           history: Optional[List[History]],
                                           model_name: str = LLM_MODEL,
                                           ) -> AsyncIterable[str]:
        callback = AsyncIteratorCallbackHandler()
        model = ChatOpenAI(
            streaming=True,
            verbose=True,
            callbacks=[callback],
            openai_api_key=llm_model_dict[model_name]["api_key"],
            openai_api_base=llm_model_dict[model_name]["api_base_url"],
            model_name=model_name,
            openai_proxy=llm_model_dict[model_name].get("openai_proxy"),
            temperature = 0.1,
            # top_p = 0.5,
            # top_k = 3,
        )

        context = "\n".join([doc.page_content for doc in docs])

        input_msg = History(role="user", content=qa_prompt).to_msg_template(False)
        chat_prompt = ChatPromptTemplate.from_messages(
            [i.to_msg_template() for i in history] + [input_msg])
        

        chain = LLMChain(prompt=chat_prompt, llm=model)

        # Begin a task that runs in the background.
        task = asyncio.create_task(wrap_done(
            chain.acall({"context": context, "question": query}),
            callback.done),
        )

        source_documents = []
        for inum, doc in enumerate(docs):
            filename = os.path.split(doc.metadata["source"])[-1]
            if local_doc_url:
                url = "file://" + doc.metadata["source"]
            else:
                parameters = urlencode({"knowledge_base_name": knowledge_base_name, "file_name":filename})
                url = f"{request.base_url}knowledge_base/download_doc?" + parameters
            text = f"""出处 [{inum + 1}] [{filename}]({url}) \n\n{doc.page_content}\n\n"""
            source_documents.append(text)

        if stream:
            async for token in callback.aiter():
                # Use server-sent-events to stream the response
                yield json.dumps({"answer": token,
                                  "docs": source_documents},
                                 ensure_ascii=False)
        else:
            answer = ""
            async for token in callback.aiter():
                answer += token
            if "转人工" in answer:
                answer = "转人工"
            yield json.dumps({"answer": answer,
                              "docs": source_documents},
                             ensure_ascii=False)

        await task

    # 智能引导
    async def base_chat_iterator(query: str,
                                history: Optional[List[History]],
                                model_name: str = LLM_MODEL,
                                ) -> AsyncIterable[str]:

        model = OpenAI(
            streaming=True,
            verbose=True,
            openai_api_key=llm_model_dict[model_name]["api_key"],
            openai_api_base=llm_model_dict[model_name]["api_base_url"],
            model_name=model_name,
            openai_proxy=llm_model_dict[model_name].get("openai_proxy"),
            temperature = 0,
            top_p = 0.5
        )

        prompt = guidance_prompt.format(instructions=instructions, user_input=query)
        input_msg = History(role="user", content=prompt).to_msg_template()
        chat_prompt = ChatPromptTemplate.from_messages(
            [i.to_msg_template() for i in history] + [input_msg])
        
        prompt = chat_prompt.format()

        response = model.invoke(prompt)

        if "转人工" in response:
            response = "转人工"

        if stream:
            for token in response:
            # Use server-sent-events to stream the response
                a = {"answer": token,
                                "docs": []}
                yield json.dumps({"answer": token,
                                "docs": []},
                                ensure_ascii=False)
        else:
            yield json.dumps({"answer": response,
                            "docs": []},
                            ensure_ascii=False)
    
    # 跑量诊断
    async def analyze_chat_iterator(query: str,
                                history: Optional[List[History]],
                                model_name: str = LLM_MODEL,
                                ) -> AsyncIterable[str]:

        model = OpenAI(
            streaming=True,
            verbose=True,
            openai_api_key=llm_model_dict[model_name]["api_key"],
            openai_api_base=llm_model_dict[model_name]["api_base_url"],
            model_name=model_name,
            openai_proxy=llm_model_dict[model_name].get("openai_proxy"),
            temperature = 0,
            top_p = 0.5
        )

        # 获取设备id
        id_list = extract_id(query)

        # 分情况处理
        response = ""
        if len(id_list) > 1:
            response += "您输入的id多于一个，现在分析第一个：{id_list[0]}。"
            id_list = id_list[:1]
        
        device_info = ""
        if len(id_list) > 0:
            device_uuid = id_list[0]
            # 若处理设备不存在，抛出异常
            device_usage_info =  DeviceUsageInfo(device_uuid)
            device_test_info = DeviceTestInfo(device_uuid)
            device_nat_info = DeviceNATInfo(device_uuid)
            device_load = DeviceLoad(device_uuid)
            task_id = DeviceTaskid(device_uuid)
            task_redline = TaskRedLine(task_id)
            change_point = ChangePointInfo(device_uuid)
            device_info = device_info_template.format(DeviceUsageInfo=device_usage_info,
                                                      DeviceTestInfo=device_test_info,
                                                      DeviceNATInfo=device_nat_info,
                                                      TaskRedLine=task_redline,
                                                      ChangePointInfo = change_point,
                                                      DeviceLoad = device_load
                                                      )
            
        prompt = analyze_prompt.format(user_input=query, device_info=device_info)

        input_msg = History(role="user", content=prompt).to_msg_template()
        chat_prompt = ChatPromptTemplate.from_messages(
            [i.to_msg_template() for i in history] + [input_msg])
        
        prompt = chat_prompt.format()

        response = model.invoke(prompt)

        if "转人工" in response:
            response = "转人工"

        if device_info != "":
            response = response + f"\n\n 附设备信息：{device_info}"
        if stream:
            for token in response:
            # Use server-sent-events to stream the response
                a = {"answer": token,
                                "docs": []}
                yield json.dumps({"answer": token,
                                "docs": []},
                                ensure_ascii=False)
        else:
            yield json.dumps({"answer": response,
                            "docs": []},
                            ensure_ascii=False)

    # 不清楚
    async def inquiry_chat_iterator(query: str,
                                history: Optional[List[History]],
                                model_name: str = LLM_MODEL,
                                ) -> AsyncIterable[str]:

        model = OpenAI(
            streaming=True,
            verbose=True,
            openai_api_key=llm_model_dict[model_name]["api_key"],
            openai_api_base=llm_model_dict[model_name]["api_base_url"],
            model_name=model_name,
            openai_proxy=llm_model_dict[model_name].get("openai_proxy"),
            temperature = 0,
            top_p = 0.5
        )

        prompt = inquiry_prompt.format(functions=functions, user_input=query)
        input_msg = History(role="user", content=prompt).to_msg_template()
        chat_prompt = ChatPromptTemplate.from_messages(
            [i.to_msg_template() for i in history] + [input_msg])
        
        prompt = chat_prompt.format()

        response = model.invoke(prompt)

        if stream:
            for token in response:
            # Use server-sent-events to stream the response
                a = {"answer": token,
                                "docs": []}
                yield json.dumps({"answer": token,
                                "docs": []},
                                ensure_ascii=False)
        else:
            yield json.dumps({"answer": response,
                            "docs": []},
                            ensure_ascii=False)

    def  chat_iterator(response):

        if stream:
            for token in response:
            # Use server-sent-events to stream the response
                a = {"answer": token,
                                "docs": []}
                yield json.dumps({"answer": token,
                                "docs": []},
                                ensure_ascii=False)
        else:
            yield json.dumps({"answer": response,
                            "docs": []},
                            ensure_ascii=False)


    if intent == Intent.qa:
        return StreamingResponse(knowledge_base_chat_iterator(query, kb, top_k, history, model_name),
                             media_type="text/event-stream")
    elif intent == Intent.guidance:
        return StreamingResponse(base_chat_iterator(query,history, model_name),
                             media_type="text/event-stream")
    elif intent == Intent.analyze:
        return StreamingResponse(analyze_chat_iterator(query,history, model_name),
                             media_type="text/event-stream")
    elif intent == Intent.unclear:
        return StreamingResponse(inquiry_chat_iterator(query, history, model_name),
                             media_type="text/event-stream")
    else:
        return StreamingResponse(chat_iterator(call_artificial),
                        media_type="text/event-stream")