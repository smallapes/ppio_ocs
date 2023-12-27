import json
import requests
from langchain.utilities import SerpAPIWrapper
from configs.ppio_config import API_TOKEN, API_URL, DEBUG, NAT_DETAIL_DEFAULT
import time
import logging
from datetime import datetime

fprefix = "智能调用"
def warn(info):
    logging.info(info)


recruit_2_taskid = {}
taskid_2_recruit = {}
start = time.time()
def get_taskid_map():
    global start
    global recruit_2_taskid, taskid_2_recruit
    headers = {'content-type': "application/json", 'Authorization': API_TOKEN}
    url = API_URL+'/metadata/v1/business_metadata/list'
    if time.time() - start > 3600 or recruit_2_taskid == {}:
        r = requests.get(url, headers=headers)  # 发送到服务端
        if r.status_code == 200:
            prams = json.loads(r.text)
            recruit_2_taskid, taskid_2_recruit = {}, {}
            for k in prams.get('businessMetadata', {}):
                recruit_2_taskid[k.get('recruitName')] = k.get('id')
                taskid_2_recruit[k.get('id')] = k.get('recruitName')
            start = time.time()
        else:
            warn(f"查询招募名出错：{r}")
    return recruit_2_taskid

get_taskid_map()
print(33, list(recruit_2_taskid), list(taskid_2_recruit))

token = 'Bearer RNlnM2mr18YRsRnv030HqBs15wMOakG53sO3j9ycOevnlkxSCD9w55JInVm0CAH9'


def get_current_weather(location):
    """Get the current weather in a given location"""
    search = SerpAPIWrapper(serpapi_api_key="5f0619a1231f66569a4f7dce9adbf8001b3a21cb6face3c3ac35d6c3dc5981dd")
    return search.run(f"{location}的今天气温如何？")
    

def get_current_population(location):
    """Get the current population in a given location"""
    search = SerpAPIWrapper(serpapi_api_key="5f0619a1231f66569a4f7dce9adbf8001b3a21cb6face3c3ac35d6c3dc5981dd")
    return search.run(f"{location}的有多少人口？")


def is_redline_matched_api_single(deviceUUID, task):
    recruit_2_taskid = get_taskid_map()
    taskid = recruit_2_taskid.get(task.lower()) or recruit_2_taskid.get(task.upper())
    if taskid == None:
        logging.info(f"错误：找不到任务 {task}，不在{recruit_2_taskid}")
        return f"错误：找不到任务 {task}"

    url = 'https://api.painet.work/device/v1/devicesBusinessRedlinePass'
    data = {}
    data['devicesBusinessRedline'] = []
    data['devicesBusinessRedline'].append({"deviceUUID": deviceUUID, "business": task})
    headers = {'content-type': "application/json", 'Authorization': token}

    try:
        r = requests.post(url, data=json.dumps(data, ensure_ascii=False).encode('utf-8'), headers=headers)  # 发送到服务端
        prams = json.loads(r.text)
        if prams['devicesBusinessRedline'][0]["pass"]:
            res =  f"{deviceUUID} 符合{task}要求，可以部署"
        else:
            res =  f"{deviceUUID} 不符合{task}要求，不能部署"
    except:
        res = "未获得结果，请检查设备号和任务！"
    return res

import traceback
def get_device_info(deviceUUID):
    data = {}
    data['deviceUuid'] = [deviceUUID]
    headers = {'content-type': "application/json", 'Authorization': API_TOKEN}
    url = API_URL+'/device/v1/info/basic'
    res = ""
    try:
        r = requests.get(url, params=data, headers=headers)  # 发送到服务端
        prams = json.loads(r.text)
        device_info = prams.get("devices", [])
        if len(device_info) > 0:
            info = device_info[0]
            test_result = info.get('deviceBandwidthTestResult', {})
            real_task = info.get('deviceTaskInfo', {}).get('realTasks', [])

            task_redline = ""
            for task_id in real_task:
                task_redline += get_task_redline(task_id)
            def pf(f):
                if f == '':
                    return f
                return f"{int(f*100)} %"
            real_task_recruit = [taskid_2_recruit.get(task_id, '') for task_id in real_task]
            res =  f"""设备信息：NAT类型， {info.get('realtimeNATTypeType', '')}，业务 {real_task_recruit}，设备昨日利用率为 {pf(info.get('bandwidthRatio', ''))}， 业务区域昨日利用率为 {pf(info.get('bandwidthRatioOfBusinessAndArea', ''))}，压测信息为 极限压测满意度 {pf(test_result.get("upbandwidthTestSatisfaction", ''))}，丢包压测满意度 {pf(test_result.get("upbandwidthTestSatisfactionWithTCP", ''))}; \n 业务红线：{task_redline};
                    """
    except:
        res = "未获得结果，请检查设备id是否正确"
        traceback.print_exc()
    return res


def get_task_redline(task_id):
    """
    task_id: 招募名或者业务名
    """
    if task_id in taskid_2_recruit:
        task_id = taskid_2_recruit.get(task_id)

    recruit_2_taskid = get_taskid_map()
    taskid = recruit_2_taskid.get(task_id.lower()) or recruit_2_taskid.get(task_id.upper())
    if taskid == None:
        logging.info(recruit_2_taskid)
        return f"错误：找不到任务 {id}"

    data = {}
    data['businessId'] = taskid
    headers = {'content-type': "application/json", 'Authorization': API_TOKEN}
    url = API_URL+'/v1/entry_standard'
    res = ""
    try:
        r = requests.get(url, params=data, headers=headers)  # 发送到服务端
        prams = json.loads(r.text)
        redline = prams.get('data', {}).get("list", [])
        if len(redline) > 0:
            natDetail = redline[0].get('natDetail')
            if natDetail is None:
                natDetail = NAT_DETAIL_DEFAULT
            nat = " 或 ".join([k for k,v in natDetail.items() if v != '' and v != [] ])
            res = f"""{task_id} 要求：NAT类型，{nat}"""
        else:
            res = f"{taskid} 未找到网路硬件要求"
    except:
        traceback.print_exc()
        res = "未获得结果，请检查任务id是否正确"

    # logging.info(res)
    return res

import matplotlib.pyplot as plt
import ruptures as rpt
import numpy as np
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def get_timestamp():
    import time
    import pytz
    from datetime import datetime

    # 设置中国时区
    china_timezone = pytz.timezone('Asia/Shanghai')

    # 获取当前时间
    current_utc_time = datetime.utcnow()

    # 将当前时间转换为中国时区
    current_china_time = current_utc_time.replace(tzinfo=pytz.utc).astimezone(china_timezone)

    # 获取当前中国时区的Unix时间戳
    china_timestamp = int(current_china_time.timestamp())

    # print(current_china_time, current_china_time.timestamp(), time.time())

    # 打印结果
    return china_timestamp # TODO（获取时区）



def get_change_point(deviceUUID, days=7):
    data = {}
    data['deviceUUID'] = deviceUUID
    data["endTime"] = str(get_timestamp())
    data["startTime"] = str(int(data["endTime"])- days*24*3600)
   
    # 跑量
    headers = {'content-type': "application/json", 'Authorization': API_TOKEN}
    url = API_URL+"/device/v1/monitor/bw"
    r = requests.get(url, params=data, headers=headers)  # 发送到服务端
    bw = json.loads(r.text)
    bws = bw.get('bws', [])

    # 延迟与重传
    headers = {'content-type': "application/json", 'Authorization': API_TOKEN}
    url = API_URL+'/device/v1/monitor/network'

    r = requests.get(url, params=data, headers=headers)  # 发送到服务端
    prams = json.loads(r.text)

    tcpRends = prams.get('tcpRends', [])
    outAvgDelays = prams.get('outAvgDelays', [])

    # cpu、内存、磁盘延迟
    url = API_URL+"/device/v1/monitor/hardware"
    r = requests.get(url, params=data, headers=headers)  # 发送到服务端
    hw = json.loads(r.text)
    cpu_usages = hw.get('cpuUsages', [])
    mem_usages = hw.get('memUsages', [])
    diskDelays = hw.get('diskDelays', [])
    disk_dict = {}
    for item in diskDelays:
        name = item.get('name')
        if disk_dict.get(name) is None:
            disk_dict[name] = []
        disk_dict[name].append(item)
    disk_list = [(k, v) for k, v in disk_dict.items()]

    # 变点检测数据
    feature_labels = ["跑量", "TCP重传", "出网延迟", "CPU利用率", "内存利用率"]
    feature_labels_en = ["bws", "tcpRends", "outAvgDelays", "cpu_usages", "mem_usages"]
    feature_lists = [tcpRends, outAvgDelays, cpu_usages, mem_usages]
    feature_units = ['G', '%', 'ms', '%', '%']
    for k, v in disk_list:
        feature_lists.append(v)
        feature_labels.append(f"磁盘延迟: {k}")
        feature_labels_en.append(f"disk delay: {k}")
        feature_units.append('ms')
    for i, feature in enumerate(feature_lists):
        feature_lists[i] = {item.get('time'): item.get('data') for item in feature}
    
    # tcpRends_dict = {item.get('time'): item.get('data') for item in tcpRends}
    # outAvgDelays_dict = {item.get('time'): item.get('data') for item in outAvgDelays}
    signal = []
    for item in bws:
        ts = item.get('time')
        bwi = item.get('data')
        row = [bwi]
        for feature in feature_lists:
            row.append(feature.get(ts, 0))
        signal.append(row)
    n_features = len(feature_labels)
    signal = np.array(signal)

    # 变点检测
    algo = rpt.Pelt(model="rbf").fit(signal)
    result = algo.predict(pen=20)

    # display
    if DEBUG:    
        bkps = result
        fig, ax = rpt.display(signal, bkps, result,  figsize= (12, 2 * n_features))
        
        for i, axi in enumerate(ax):
            axi.set_ylabel(f"{feature_labels_en[i]} {feature_units[i]}")
            
        plt.show()
        plt.savefig("./ruptures.png")
    
    # 获取跑量降低点（相对降低)
    # 规则，跑量(bws)下降，且低于过去自然日区间平均水平
    intervals = [0] + result
    if intervals[-1] < len(bws) -1:
        intervals.append(len(bws) -1)
    new_result = []
    bws_dict = {item.get('time'): item.get('data') for item in bws}
    for i in range(1, len(intervals)-1):
        if intervals[i+1] >= len(bws):
            continue
        interval1 = (intervals[i-1], intervals[i])
        interval2 = (intervals[i], intervals[i+1])

        area1 = signal[interval1[0]:interval1[1], 0]
        area2 = signal[interval2[0]:interval2[1], 0]
        change_ratio = (sum(area2) / len(area2) - sum(area1) / len(area1)) / (sum(area1) / len(area1))

        # 同比
        starttime = bws[intervals[i]].get('time')
        endtime = bws[intervals[i+1]].get('time')
        # starthm = datetime.fromtimestamp(int(starttime)).strftime("%H:%M:%S")
        # endhm = datetime.fromtimestamp(int(endtime)).strftime("%H:%M:%S")
        starthm = int(starttime)
        endhm = int(endtime)

        temp = []
        base = []
        for item in bws:
            time = item.get('time', starttime)
            timehm = int(time)
            for j in range(1, days):
                new = timehm + j * 24 * 3600
                if new > starthm and new < endhm:
                    temp.append(item.get('data', 0))
                    base.append(bws_dict.get(str(int(new)), item.get('data', 0)))
        if len(temp) == 0:
            continue
        change_ratio_2 = (sum(base) / len(base) - sum(temp) / len(temp)) / (sum(temp) / len(temp))
        if change_ratio < - 0.5 and change_ratio_2 < - 0.5:
            new_result.append(intervals[i])
            print('change_ratio',  change_ratio, change_ratio_2)



    # 处理后 display
    if DEBUG:    
        fig, ax = rpt.display(signal, new_result, new_result,  figsize= (12, 2 * n_features))
        for i, axi in enumerate(ax):
            axi.set_ylabel(f"{feature_labels_en[i]} {feature_units[i]}")
            plt.show()
            plt.savefig("./ruptures_postprocess.png")

    # 收集信息
    res = ""
    for i, bkp in enumerate(new_result):
        bkp = min(len(bws)-1, bkp + 5)
        info = signal[bkp, :]
        info[0] = info[0]/1024**3
        ts = bws[bkp].get('time')
        dt = datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M:%S")

        name_info = [f"{k}为{v:.2f} {u}" for k, v, u in zip(feature_labels, info, feature_units)]
        res += f"异常点 {i + 1}，时间 {dt}， {'，'.join(name_info)}；\n"
    return res


def get_demand(location='', isp='', natDetail='', upbandwidthPerLine='', upbandwidth='', networkType='', diskType='', deviceUUID=''):
    data = {'location': location, 'isp': isp, 'natDetail': natDetail, 'upbandwidthPerLine': upbandwidthPerLine, 
            'upbandwidth': upbandwidth, 'networkType': networkType, "diskType": diskType, 'deviceUUID': deviceUUID}
    fprefix = "看需求"
    headers = {'content-type': "application/json", 'Authorization': API_TOKEN}
    url = API_URL+"/v1/user/operating_requirement"
    r = requests.get(url, params=data, headers=headers)  # 发送到服务端
    demand = json.loads(r.text)
    res = ''
    if r.status_code == 200:
        logging.info(f"{fprefix}: {r.json()}")
        data_list = demand.get('list', [])
        for item in data_list:
            recruit_name = item.get('recruitName')
            requires = item.get("requires", [])
            for rl in requires:
                region = rl.get('region')
                location = rl.get('location')
                ispGap = rl.get('ispGap')
                ispGap = sorted([(k,v) for k, v in ispGap.items() if ( v != 0 and k in isp ) or isp == ''], key=lambda x: x[0])
                if len(ispGap) > 0:
                    res += f"{location if location else region}，{'、'.join([f'{k}' for k, v in ispGap]) }，{recruit_name} 有需求；\n"
    else:
        print(r)
        print(r.json())
    if res == '':
        res = "无需求"
    return res


def get_acceptance(deviceUUID=''):
    data = {'deviceUuid': deviceUUID}
    headers = {'content-type': "application/json", 'Authorization': API_TOKEN}
    url = API_URL+'/device/v1/info/basic'
    r = requests.get(url, params=data, headers=headers)  # 发送到服务端

    res = f"{deviceUUID} 验收中"
    if r.status_code == 200:
        info = json.loads(r.text)
        devices = info.get("devices", [])
        if len(devices) > 0:
            device = devices[0]
            taskInfo = device.get("deviceTaskInfo", {})
            specificTasks = taskInfo.get("specificTasks", [])
            realTasks = taskInfo.get("realTasks", [])
            zjDeployState = taskInfo.get('zjDeployState', '')

            recruit_name = ",".join([taskid_2_recruit.get(task_id) for task_id in realTasks if taskid_2_recruit.get(task_id) is not None])
            if len(realTasks) == 1 and realTasks[0] == 'zjtd' and zjDeployState == "验收通过":
                res = f"{deviceUUID}  验收业务 {recruit_name} 成功"
            elif len(set(specificTasks)) > 0 and set(realTasks) == set(specificTasks):
                res = f" {deviceUUID} 验收业务 {recruit_name} 成功"
    return res


def get_recommend(deviceUUID=''):
    headers = {'content-type': "application/json"}
    url = 'http://42.192.6.240:5005/http/query/'
    """
    {"request_type":5,"redline_filter_type":1,"task_num":1,
    "sub_request_param":[{"device_uuid":"bd52acc9efaae1d74322eb67dfbb044a","task_id":"ksvb"}]
    """
    data = {
            "request_type":1,
            "redline_filter_type":1,
            "task_num":1,
            'sub_request_param':
                [{'device_uuid': deviceUUID}]
            }
    r = requests.post(url, data = json.dumps(data, ensure_ascii=False).encode('utf-8'), headers=headers)  # 发送到服务端

    res = f'{deviceUUID}暂无推荐任务。'
    if r.status_code == 200:
        info = json.loads(r.text)
        tasks = []
        for task in info:
            specific_tasks = task.get('specific_tasks', '')
            sts = specific_tasks.split(",")
            rns = [taskid_2_recruit.get(taskid, taskid) for taskid in sts]
            rn = ','.join(rns)
            sort_value = task.get('use_rate_dif', 0)
            tasks.append([rn, sort_value])
        sort_tasks = sorted(tasks, key = lambda x: x[1])
        sort_tasks = [f"推荐{i+1}，{item[0]}"for i, item in enumerate(sort_tasks)]
        if len(sort_tasks) > 0:
            res = f'{deviceUUID} 推荐任务：{";".join(sort_tasks)}。'
    return res


# ---------------------- 函数调用 ----------------------
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "查询一个位置的当前的气温",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    }
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_population",
            "description": "查询一个位置的当前的人口数",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "省市、县等行政规划，黄山等自然规划, 例如、北京",
                    }
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "is_redline_matched_api_single",
            "description": "判断xxx设备能够部署xxx业务，符不符合业务的要求; 常见的问法： xxx能上xx、xxx部署xx、xxx yy、xxx切yy",
            "parameters": {
                "type": "object",
                "properties": {
                    "deviceUUID": {
                        "type": "string",
                        "description": "设备号，比较长，例如3f725d53e56f494d9cab9f50e6b6ffe3",
                    },
                    "task": {
                        "type": "string",
                        "description": "任务名，比较短，比如长a，短b，精品x",
                        "enum": list(recruit_2_taskid)
                    },
                },
                "required": ["deviceUUID", "task"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_device_info",
            "description": "查看xxxxx设备的NAT类型， 业务，昨日利用率，设备业务的区域昨日利用率，压测满意度",
            "parameters": {
                "type": "object",
                "properties": {
                    "deviceUUID": {
                        "type": "string",
                        "description": "设备号，比较长，例如3f725d53e56f494d9cab9f50e6b6ffe3",
                    },
                },
                "required": ["deviceUUID"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_task_redline",
            "description": "查看业务的硬件、网络要求",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "任务名，比较短，比如长a，短b，精品x",
                        "enum": list(recruit_2_taskid)
                    },
                },
                "required": ["task_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_acceptance",
            "description": "查看xxxxx设备的验收状态, xxxxxx 验收，xxxxxx 设备验收",
            "parameters": {
                "type": "object",
                "properties": {
                    "deviceUUID": {
                        "type": "string",
                        "description": "设备号，比较长，例如3f725d53e56f494d9cab9f50e6b6ffe3",
                    },
                },
                "required": ["deviceUUID"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recommend",
            "description": "xxxx推荐任务，xxxx切换推荐，xxxx可以部署啥，xxxx可以切啥",
            "parameters": {
                "type": "object",
                "properties": {
                    "deviceUUID": {
                        "type": "string",
                        "description": "设备号，比较长，例如3f725d53e56f494d9cab9f50e6b6ffe3",
                    },
                },
                "required": ["deviceUUID"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_demand",
            "description": "查看某个地区的业务需求, 例如查看北京电信需求、上海联通 需求",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "地点名，省份，直辖市，例如湖北、上海",
                    },
                    "isp": {
                        "type": "string",
                        "description": "运营商, 例如 '电信', '移动', '联通'， 如果有多个，','隔开，例如 '电信,联通'",
                    },
                    "natDetail": {
                        "type": "string",
                        "description": "nat类型，NAT0、NAT1、NAT2、NAT3、NAT4，如果有多个，','隔开，例如 'NAT0,NAT1'",
                    },
                    "upbandwidthPerLine": {
                        "type": "integer",
                        "description": "当条上行 单位：B, 如果是M，需要*1024*1024转化成B"
                    },
                    "upbandwidth": {
                        "type": "integer",
                        "description": "当机上行 单位：B, 如果是M，需要*1024*1024转化成B"
                    },
                    "diskType": {
                        "type": "string",
                        "description": "磁盘类型 HDD：hdd，SSD：ssd 混盘(HDD+SSD)：md，",
                        "enum": ['hdd', 'ssd', 'md']
                    },
                    "deviceUUID": {
                        "type": "string",
                        "description": "设备号，比较长，例如3f725d53e56f494d9cab9f50e6b6ffe3",
                    },
                },
            },
        },
    }
]

available_functions = {
    "get_current_weather": get_current_weather,
    "get_current_population": get_current_population,
    "is_redline_matched_api_single": is_redline_matched_api_single,
    "get_device_info": get_device_info,
    "get_task_redline": get_task_redline,
    'get_demand': get_demand,
    'get_acceptance': get_acceptance,
    'get_recommend': get_recommend,
}  # only one function in this example, but you can have multiple


def run_conversation(query = "What's the pulation like in San Francisco, Tokyo, and Paris?", history=[], model_name="gpt-3.5-turbo-1106"):
    from openai import OpenAI

    client = OpenAI(
            api_key="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",
            base_url="http://45.116.14.16:3001/proxy/v1",
            # model_name="gpt-3.5-turbo",
            )

    # Step 1: send the conversation and available functions to the model
    messages = [{"role": "user", "content": query}]
    if len(history) > 0:
        h = history[-1]
        messages.append({'role': h.role, 'content': h.content})
    logging.info(f"{fprefix}: {messages}")
    response = client.chat.completions.create(
        model=model_name, # TODO (猫仙人)
        messages=messages,
        tools=tools,
        tool_choice="auto",  # auto is default, but we'll be explicit
    )
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    # Step 2: check if the model wanted to call a function
    if tool_calls:
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors

        messages.append(response_message)  # extend conversation with assistant's reply
        # Step 4: send the info for each function call and function response to the model
        responses = []
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            logging.info(f"{fprefix}: {function_name}")
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(
                **function_args
                # location=function_args.get("location"),
                # unit=function_args.get("unit"),
            )
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )  # extend conversation with function response
            responses.append(function_response)
            logging.info(f"{fprefix}: {function_name}, {function_response}")

        # if function_name not in ['get_current_weather', "get_current_population"]:
        #     return '\n'.join(responses)
        # if function_name in ["get_demand", "get_recommend", "is_redline_matched_api_single", "get_acceptance", "get_task_redline"]:
        #     return '\n'.join(responses)
        
        second_response = client.chat.completions.create(
            model=model_name,
            messages=messages,
        )  # get a new response from the model where it can see the function response
        res =  second_response.choices[0].message.content
        if type(res) != str:
            res = '未获取到结果'

    else:
        res = response_message.content
    logging.info(f"{fprefix}: {res}")
    return res


if __name__ == "__main__":
    ## agent 测试
 

    # 业务红线
    # print(run_conversation("长b啥要求"))
    # print(get_task_redline('长视频b'))

    ## 设备信息
    # print(run_conversation('d2f09d14cb6240b7bfb0d59f322090df NAT类型'))
    # print(run_conversation('d2f09d14cb6240b7bfb0d59f322090df 是啥NAT类型'))
    # print(get_device_info('22eded319d9a365ba82d1477a1ba6b06'))



    # **跑量低1/5**
    # print(run_conversation("22eded319d9a365ba82d1477a1ba6b06 跑量低"))

    # 变点
    # print(get_change_point('22eded319d9a365ba82d1477a1ba6b06'))
    # get_timestamp() 

    # **查验收2/5**
    # print(get_acceptance('22eded319d9a365ba82d1477a1ba6b06'))
    # print(get_acceptance('4d62ee0501bbd8d96f9c6d857c3e2737'))
    # print(run_conversation("22eded319d9a365ba82d1477a1ba6b06 验收"))

    # **查需求3/5**
    # print(get_demand())
    # print(run_conversation("上海 电信 联通 需求"))
    # print(run_conversation("广东 需求"))


    # **切换推荐4/5**
    # print(get_recommend("120088f7eb5115c4a48ac8c72ef1b111"))
    # print(run_conversation("120088f7eb5115c4a48ac8c72ef1b111可以切啥"))
    # print(run_conversation("d2f09d14cb6240b7bfb0d59f322090df可以切啥"))
    # device_uuids = ["120088f7eb5115c4a48ac8c72ef1b111"]
    # for id in device_uuids:
    #     print(get_recommend(id))
    #     print(run_conversation(f"{id} 推荐"))

    # **是否符合红线5/5**
    # print(run_conversation("4d62ee0501bbd8d96f9c6d857c3e2737 能上长b吗"))

    print(run_conversation("你好"))