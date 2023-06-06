# coding=utf-8
# Copyright 2023 South China University of Technology and 
# Engineering Research Ceter of Ministry of Education on Human Body Perception.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Author: Chen Yirong <eeyirongchen@mail.scut.edu.cn>
# Date: 2023.06.06

''' 运行方式
```bash
pip install streamlit # 第一次运行需要安装streamlit
pip install streamlit_chat # 第一次运行需要安装streamlit_chat
streamlit run soulchat_app.py --server.port 9026
```
## 测试访问
http://<your_ip>:9026

'''

import os
import re
import json
import torch
import streamlit as st
from streamlit_chat import message
from transformers import AutoModel, AutoTokenizer

# st-chat uses https://www.dicebear.com/styles for the avatar
# https://emoji6.com/emojiall/

# 指定显卡进行推理
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 下载模型并且指定路径
model_name_or_path = 'scutcyr/SoulChat'

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)


def answer(user_history, bot_history, sample=True, top_p=0.75, temperature=0.95):
    '''sample：是否抽样。生成任务，可以设置为True;
    top_p：0-1之间，生成的内容越多样
    max_new_tokens=512 lost...'''

    if len(bot_history)>0:
        dialog_turn = 5 # 设置历史对话轮数
        if len(bot_history)>dialog_turn:
            bot_history = bot_history[-dialog_turn:]
            user_history = user_history[-(dialog_turn+1):]
        
        context = "\n".join([f"用户：{user_history[i]}\n心理咨询师：{bot_history[i]}" for i in range(len(bot_history))])
        input_text = context + "\n用户：" + user_history[-1] + "\n心理咨询师："
    else:
        input_text = "用户：" + user_history[-1] + "\n心理咨询师："
        return "你好！我是你的个人专属数字辅导员甜心老师，欢迎找我倾诉、谈心，期待帮助到你！"
    
    print(input_text)
    if not sample:
        response, history = model.chat(tokenizer, query=input_text, history=None, max_length=2048, num_beams=1, do_sample=False, top_p=top_p, temperature=temperature, logits_processor=None)
    else:
        response, history = model.chat(tokenizer, query=input_text, history=None, max_length=2048, num_beams=1, do_sample=True, top_p=top_p, temperature=temperature, logits_processor=None)

    print("模型原始输出：\n", response)
    # 规则校验，这里可以增加校验文本的规则
    response = re.sub("\n+", "\n", response)
    print('心理咨询师: '+response)
    return response
    

st.set_page_config(
    page_title="SoulChat(内测版)",
    page_icon="👩‍🏫",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': """     
-   版本：👩‍🏫SoulChat(内测版)
-   机构：广东省数字孪生人重点实验室
-   作者：陈艺荣
	    """
    }
)

st.header("👩‍🏫SoulChat(内测版)")

with st.expander("ℹ️ - 关于我们", expanded=False):
    st.write(
        """     
-   版本：👩‍🏫SoulChat(内测版)
-   机构：广东省数字孪生人重点实验室
-   作者：陈艺荣
	    """
    )

# https://docs.streamlit.io/library/api-reference/performance/st.cache_resource
@st.cache_resource
def load_model():
    model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True).half()
    model.to(device)
    print('Model Load done!')
    return model

@st.cache_resource
def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    print('Tokenizer Load done!')
    return tokenizer

model = load_model()
tokenizer = load_tokenizer()


if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

user_col, ensure_col = st.columns([5, 1])

def get_text():
    input_text = user_col.text_area("请在下列文本框输入您的咨询内容：","", key="input", placeholder="请输入您的求助内容，并且点击Ctrl+Enter(或者发送按钮)确认内容")
    if ensure_col.button("发送", use_container_width=True):
        if input_text:
            return input_text  
    else:
        if input_text:
            return input_text

user_input = get_text()

if 'id' not in st.session_state:
    if not os.path.exists("./history"):
        # 创建保存用户聊天记录的目录
        os.makedirs("./history")
    json_files = os.listdir("./history")
    id = len(json_files)
    st.session_state['id'] = id

if user_input:
    st.session_state.past.append(user_input)
    output = answer(st.session_state['past'],st.session_state["generated"])
    st.session_state.generated.append(output)
    #bot_history.append(output)
    # 将对话历史保存成json文件
    dialog_history = {
        'user': st.session_state['past'],
        'bot': st.session_state["generated"]
    }
    with open(os.path.join("./history", str(st.session_state['id'])+'.json'), "w", encoding="utf-8") as f:
        json.dump(dialog_history, f, indent=4, ensure_ascii=False)

if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])):
        if i == 0:
            # 
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user', avatar_style="avataaars", seed=26)
            message("你好！我是你的个人专属数字辅导员甜心老师，欢迎找我倾诉、谈心❤️，期待帮助到你！🤝🤝🤝"+"\n\n------------------\n以下回答由灵心大模型SoulChat自动生成，仅供参考！", key=str(i), avatar_style="avataaars", seed=5)
        else:
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user', avatar_style="avataaars", seed=26)
            #message(st.session_state["generated"][i]+"\n\n------------------\n本回答由灵心大模型SoulChat自动生成，仅供参考！", key=str(i), avatar_style="avataaars", seed=5)
            message(st.session_state["generated"][i], key=str(i), avatar_style="avataaars", seed=5)

if st.button("清理对话缓存"):
    st.session_state['generated'] = []
    st.session_state['past'] = []