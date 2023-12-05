import re


def DeviceUsageInfo(device_uuid):
    return "业务区域利用率: 0.53, 设备昨日利用率0.27"

def DeviceTaskid(device_uuid):
    return "zjtd"

def DeviceTestInfo(device_uuid):
    return "IPv4：极限压测满意度: 99.2%, 丢包压测满意度：99.2%。IPv4：极限压测满意度: 0%, 丢包压测满意度：0%。"

def DeviceNATInfo(device_uuid):
    return "NAT类型：NAT0"

def ChangePointInfo(device_uuid):
    return "12月9日 12：30"

def DeviceLoad(device_uuid):
    return "磁盘延迟，200ms cpu利用率： 99%"

def TaskRedLine(task_id):
    return "NAT类型：NAT0"

