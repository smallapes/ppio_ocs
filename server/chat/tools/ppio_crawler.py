import requests
import kuser_agent
import json
from bs4 import BeautifulSoup
import html2text
import re
from urllib.parse import urlparse
from configs.server_config import API_PORT
from server.chat.prompts import platform
from configs.ppio_config import PAINET_DOC_KB_NAME, SHNAYANYUN_DOC_KB_NAME
from configs.model_config import CHUNK_SIZE


url = 'https://www.shanyancloud.work/api/help/v1'

def get_website(url, website="派享云"):
    tag_url = f"{url}/tags"
    article_base = f'{url}/article'

    html = requests.get(url=tag_url,headers={'User-Agent':kuser_agent.get()})

    help_center = json.loads(html.text)

    tags = help_center.get('tags', [])


    def get_article(url):
        html = requests.get(url=url, headers={"User-Agent": kuser_agent.get()}).text
        html = json.loads(html).get('article',{}).get('content', '')

        bs = BeautifulSoup(html)
        text = bs.prettify()

        ht = html2text.HTML2Text()
        ht.bypass_tables=False
        ht.mark_code = True
        ht.code = True

        result = ht.handle(text)
        return result


    for tag in tags:
        name = tag.get('name', '')
        articles = tag.get('articles', [])
        for article in articles:
            id = article.get('id')
            title = article.get('title')
            a_url = f"{article_base}/{id}"
            content = get_article(a_url)
            out_file_name = f"/home/aitest2/maoxianren/langchain-chatchat-dev/server/chat/data/{website}_{name}_{title}.txt"
            with open(out_file_name, 'w') as ofile:
                ofile.write(f"**{name}: {title}**\n\n")
                ofile.write(content)
                print(f"保存到{out_file_name}")



def find_links(soup, domain):
    links = soup.find_all('a', href=lambda href: href and domain in href)
    return [link['href'] for link in links]


def get_feishu_page(depth, total, name, base_url: str, domain: str):
    # 1. 获取整个页面数据
    html = requests.get(url=base_url,headers={'User-Agent':kuser_agent.get()}).content
    # 2. 提取正文中的HTML文本
    bs = BeautifulSoup(html)
    text = bs.prettify()

    # 3. 将HTML文本转成Markdown格式
    ht = html2text.HTML2Text()
    # 一般的参数设置
    ht.bypass_tables = False
    ht.mark_code = True
    ht.code = True
    # --------------
    result = ht.handle(text)
    if depth > 0:
        links = find_links(bs, domain)
        for link in links:
            print(78, link)
            get_feishu_page(depth-1, total, name, link, domain)

    out_file_name = f"/home/aitest2/maoxianren/langchain-chatchat-dev/server/chat/data/{name}_{total[0]}.txt"
    with open(out_file_name, 'w') as ofile:
        ofile.write(result)
        print(f"保存到{out_file_name}")
    total[0] += 1



def get_feishu(url="https://ppio-cloud.feishu.cn/docx/HKb5dzzACoWiN1xhMlEckOmfn5b", name="飞书_闪燕云"):
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    total = [0]
    get_feishu_page(5, total, name, url, domain)
    


def upload_directory(target_directory='/home/shandian/doc_data/ppio_ocs',  
                     knowledge_base_name="ppio_ocs_painet"):
    import os
    import glob
    import requests
    import shutil
    import json
    import time

    #知识库名称

    url = f'http://0.0.0.0:{API_PORT}/knowledge_base/upload_doc'

    #获取所有文件的绝对路径
    all_files = glob.glob(os.path.join(
        target_directory, '**/*.*'), recursive=True)

    headers = {
        'accept': 'application/json'
    }
    data = {
        'knowledge_base_name': knowledge_base_name,
        'not_refresh_vs_cache': 'false',
        "override": True
    }
    for up_file in all_files:
        #获取文件后缀
        file_extension = os.path.splitext(up_file)[1]
        if file_extension == ".docx":
            r_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        else:
            r_type='text/plain'

        files = {
            'file': (os.path.basename(up_file), open(up_file, 'rb'), r_type)
        }
        response = requests.post(url, headers=headers, data=data, files=files)
        if response.status_code == 200:
            print(response)
            print(f"上传文件{up_file}到知识库{knowledge_base_name}")
        time.sleep(0.5)



def manage_kb(name="PPIO_OCS_PAINET"):
    response = requests.get(f'http://0.0.0.0:{API_PORT}/knowledge_base/list_files?knowledge_base_name={name}')

    files = []
    if response.status_code == 200:
        res = response.json()
        files = res.get("data", [])
        print(files)


    # # 删除
    # url = f'http://0.0.0.0:{API_PORT}/knowledge_base/delete_doc'
    # headers = {
    #     "accept": "application/json",
    #     # "Content-Type": "application/json",
    # }
    # for file in files:
    #     data = {
    #         "knowledge_base_name": name,
    #         "doc_name": file,
    #         "delete_content": True,
    #         "not_refresh_vs_cache": False,
    #     }
    #     if file.startswith("派享云_") or file.startswith("闪燕云_")  or file.startswith("飞书_"):
    #         response = requests.post(url, headers=headers, json=data)
    #         if response.status_code == 200:
    #             print(f"从知识库{name}删除文件: {file}")
    # print(171, name)

    # 上传
    upload_directory(target_directory='/home/aitest2/maoxianren/langchain-chatchat-dev/server/chat/data', knowledge_base_name=name)

def main():
    kb_name = PAINET_DOC_KB_NAME

    CHUNK_SIZE = 250 # 设置
    from configs.model_config import CHUNK_SIZE as csize
    print(f"CHUNK_SIZE 设置为 {csize}")
    if platform == "派享云":
        get_website('https://www.painet.work/api/help/v1', '官网_派享云')
        get_feishu(url="https://ppio-cloud.feishu.cn/docx/doxcna94u9iYLhco0sevTmdMXsh", name="飞书_派享云")
    elif platform == "闪燕云":
        get_website('https://www.shanyancloud.work/api/help/v1', '官网_闪燕云')
        get_feishu(url="https://ppio-cloud.feishu.cn/docx/HKb5dzzACoWiN1xhMlEckOmfn5b", name="飞书_闪燕云")
        kb_name = SHNAYANYUN_DOC_KB_NAME
    else:
        raise ValueError()

    manage_kb(kb_name)
    CHUNK_SIZE = 250 # 恢复
        

if __name__ == "__main__":
    main()


