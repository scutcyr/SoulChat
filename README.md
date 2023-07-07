# [灵心（SoulChat）]((https://github.com/scutcyr/SoulChat))
<p align="center">
    <img src="./ProactiveHealthGPT.png" width=900px/>
</p>
<p align="center">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-red.svg"></a>
    <a href="support os"><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.8+-aff.svg"></a>
    <a href="https://github.com/scutcyr/SoulChat/graphs/contributors"><img src="https://img.shields.io/github/contributors/scutcyr/SoulChat?color=9ea"></a>
    <a href="https://github.com/scutcyr/SoulChat/commits"><img src="https://img.shields.io/github/commit-activity/m/scutcyr/SoulChat?color=3af"></a>
    <a href="https://github.com/scutcyr/SoulChat/issues"><img src="https://img.shields.io/github/issues/scutcyr/SoulChat?color=9cc"></a>
    <a href="https://github.com/scutcyr/SoulChat/stargazers"><img src="https://img.shields.io/github/stars/scutcyr/SoulChat?color=ccf"></a>
</p>

基于主动健康的主动性、预防性、精确性、个性化、共建共享、自律性六大特征，华南理工大学未来技术学院-广东省数字孪生人重点实验室开源了中文领域生活空间主动健康大模型基座ProactiveHealthGPT，包括：
* 经过千万规模中文健康对话数据指令微调的[生活空间健康大模型扁鹊（BianQue）](https://github.com/scutcyr/BianQue)    
* 经过百万规模心理咨询领域中文长文本指令与多轮共情对话数据联合指令微调的[心理健康大模型灵心（SoulChat）](https://github.com/scutcyr/SoulChat)   

我们期望，**生活空间主动健康大模型基座ProactiveHealthGPT** 可以帮助学术界加速大模型在慢性病、心理咨询等主动健康领域的研究与应用。本项目为 **心理健康大模型灵心（SoulChat）** 。


## 最近更新
- 👏🏻  2023.07.07: 心理健康大模型灵心（SoulChat）在线内测版本启用，欢迎点击链接使用：[灵心内测版](https://soulchat.iai007.cloud/)。
- 👏🏻  2023.06.24: 本项目被收录到[中国大模型列表](https://github.com/wgwang/LLMs-In-China)，为国内首个开源的具备共情与倾听能力的心理领域大模型。
- 👏🏻  2023.06.06: 扁鹊-2.0模型开源，详情见[BianQue-2.0](https://huggingface.co/scutcyr/BianQue-2)。
- 👏🏻  2023.06.06: 具备共情与倾听能力的灵心健康大模型SoulChat发布，详情见：[灵心健康大模型SoulChat：通过长文本咨询指令与多轮共情对话数据集的混合微调，提升大模型的“共情”能力 ](https://huggingface.co/scutcyr/SoulChat)。
- 👏🏻  2023.04.22: 基于扁鹊-1.0模型的医疗问答系统Demo，详情访问：[https://huggingface.co/spaces/scutcyr/BianQue](https://huggingface.co/spaces/scutcyr/BianQue)
- 👏🏻  2023.04.22: 扁鹊-1.0版本模型发布，详情见：[扁鹊-1.0：通过混合指令和多轮医生问询数据集的微调，提高医疗聊天模型的“问”能力（BianQue-1.0: Improving the "Question" Ability of Medical Chat Model through finetuning with Hybrid Instructions and Multi-turn Doctor QA Datasets）](https://huggingface.co/scutcyr/BianQue-1.0)


## 简介
   我们调研了当前常见的心理咨询平台，发现，用户寻求在线心理帮助时，通常需要进行较长篇幅地进行自我描述，然后提供帮助的心理咨询师同样地提供长篇幅的回复（见[figure/single_turn.png](./figure/single_turn.png)），缺失了一个渐进式的倾诉过程。但是，在实际的心理咨询过程当中，用户和心理咨询师之间会存在多轮次的沟通过程，在该过程当中，心理咨询师会引导用户进行倾诉，并且提供共情，例如：“非常棒”、“我理解你的感受”、“当然可以”等等（见下图）。
<p align="center">
    <img src="./figure/multi_turn.png" width=900px/>
</p>

   考虑到当前十分欠缺多轮共情对话数据集，我们一方面，构建了超过15万规模的 **单轮长文本心理咨询指令与答案（SoulChatCorpus-single_turn）** ，回答数量超过50万（指令数是当前的常见的心理咨询数据集 [PsyQA](https://github.com/thu-coai/PsyQA) 的6.7倍），并利用ChatGPT与GPT4，生成总共约100万轮次的 **多轮回答数据（SoulChatCorpus-multi_turn）** 。特别地，我们在预实验中发现，纯单轮长本文驱动的心理咨询模型会产生让用户感到厌烦的文本长度，而且不具备引导用户倾诉的能力，纯多轮心理咨询对话数据驱动的心理咨询模型则弱化了模型的建议能力，因此，我们混合SoulChatCorpus-single_turn和SoulChatCorpus-multi_turn构造成超过120万个样本的 **单轮与多轮混合的共情对话数据集SoulChatCorpus** 。所有数据采用“用户：xxx\n心理咨询师：xxx\n用户：xxx\n心理咨询师：”的形式统一为一种指令格式。

我们选择了 [ChatGLM-6B](https://huggingface.co/THUDM/chatglm-6b) 作为初始化模型，进行了**全量参数的指令微调**，旨在提升模型的共情能力、引导用户倾诉能力以及提供合理建议的能力。更多训练细节请留意我们后续发布的论文。

## 使用方法
* 克隆本项目
```bash
cd ~
git clone https://github.com/scutcyr/SoulChat.git
```

* 安装依赖    
需要注意的是torch的版本需要根据你的服务器实际的cuda版本选择，详情参考[pytorch安装指南](https://pytorch.org/get-started/previous-versions/)
```bash
cd SoulChat
conda env create -n proactivehealthgpt_py38 --file proactivehealthgpt_py38.yml
conda activate proactivehealthgpt_py38

pip install cpm_kernels
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

* 【补充】Windows下的用户推荐参考如下流程配置环境
```bash
cd BianQue
conda create -n proactivehealthgpt_py38 python=3.8
conda activate proactivehealthgpt_py38
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
pip install rouge_chinese nltk jieba datasets
# 以下安装为了运行demo
pip install streamlit
pip install streamlit_chat
```
* 【补充】Windows下配置CUDA-11.6：[下载并且安装CUDA-11.6](https://developer.nvidia.com/cuda-11-6-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local)、[下载cudnn-8.4.0，解压并且复制其中的文件到CUDA-11.6对应的路径](https://developer.nvidia.com/compute/cudnn/secure/8.4.0/local_installers/11.6/cudnn-windows-x86_64-8.4.0.27_cuda11.6-archive.zip)，参考：[win11下利用conda进行pytorch安装-cuda11.6-泛用安装思路](https://blog.csdn.net/qq_34740266/article/details/129137794)

* 在Python当中调用SoulChat模型    
```python
import torch
from transformers import AutoModel, AutoTokenizer
# GPU设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 加载模型与tokenizer
model_name_or_path = 'scutcyr/SoulChat'
model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True).half()
model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

# 单轮对话调用模型的chat函数
user_input = "我失恋了，好难受！"
input_text = "用户：" + user_input + "\n心理咨询师："
response, history = model.chat(tokenizer, query=input_text, history=None, max_length=2048, num_beams=1, do_sample=True, top_p=0.75, temperature=0.95, logits_processor=None)

# 多轮对话调用模型的chat函数
# 注意：本项目使用"\n用户："和"\n心理咨询师："划分不同轮次的对话历史
# 注意：user_history比bot_history的长度多1
user_history = ['你好，老师', '我女朋友跟我分手了，感觉好难受']
bot_history = ['你好！我是你的个人专属数字辅导员甜心老师，欢迎找我倾诉、谈心，期待帮助到你！']
# 拼接对话历史
context = "\n".join([f"用户：{user_history[i]}\n心理咨询师：{bot_history[i]}" for i in range(len(bot_history))])
input_text = context + "\n用户：" + user_history[-1] + "\n心理咨询师："

response, history = model.chat(tokenizer, query=input_text, history=None, max_length=2048, num_beams=1, do_sample=True, top_p=0.75, temperature=0.95, logits_processor=None)
```


* 启动服务   

本项目提供了[soulchat_app.py](./soulchat_app.py)作为SoulChat模型的使用示例，通过以下命令即可开启服务，然后，通过http://<your_ip>:9026访问。
```bash
streamlit run soulchat_app.py --server.port 9026
```
特别地，在[soulchat_app.py](./soulchat_app.py)当中，
可以修改以下代码更换指定的显卡：
```python
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
```
**对于Windows单显卡用户，需要修改为：```os.environ['CUDA_VISIBLE_DEVICES'] = '0'```，否则会报错！**

可以通过更改以下代码指定模型路径为本地路径：
```python
model_name_or_path = 'scutcyr/SoulChat'
```


## 示例
* 样例1：失恋
*
<p align="center">
    <img src="./figure/example_shilian.png" width=600px/>
</p>

* 样例2：宿舍关系

<p align="center">
    <img src="./figure/example_sushe.png" width=600px/>
</p>

* 样例3：期末考试

<p align="center">
    <img src="./figure/example_kaoshi.png" width=600px/>
</p>

* 样例4：科研压力

<p align="center">
    <img src="./figure/example_keyan.png" width=600px/>
</p>

## 声明
* 本项目使用了ChatGLM-6B 模型的权重，需要遵循其[MODEL_LICENSE](https://github.com/THUDM/ChatGLM-6B/blob/main/MODEL_LICENSE)，因此，**本项目仅可用于您的非商业研究目的**。
* 本项目提供的SoulChat模型致力于提升大模型的共情对话与倾听能力，然而，模型的输出文本具有一定的随机性，当其作为一个倾听者的时候，是合适的，但是不建议将SoulChat模型的输出文本替代心理医生等的诊断、建议。本项目不保证模型输出的文本完全适合于用户，用户在使用本模型时需要承担其带来的所有风险！
* 您不得出于任何商业、军事或非法目的使用、复制、修改、合并、发布、分发、复制或创建SoulChat模型的全部或部分衍生作品。
* 您不得利用SoulChat模型从事任何危害国家安全和国家统一、危害社会公共利益、侵犯人身权益的行为。
* 您在使用SoulChat模型时应知悉，其不能替代医生、心理医生等专业人士，不应过度依赖、服从、相信模型的输出，不能长期沉迷于与SoulChat模型聊天。

## 致谢
本项目由[华南理工大学未来技术学院](https://www2.scut.edu.cn/ft/main.htm) 广东省数字孪生人重点实验室发起，得到了华南理工大学信息网络工程研究中心、电子与信息学院等学院部门的支撑，同时致谢广东省妇幼保健院、广州市妇女儿童医疗中心、中山大学附属第三医院、合肥综合性国家科学中心人工智能研究院等合作单位。

同时，我们感谢以下媒体或公众号对本项目的报道（排名不分先后）：
* 媒体报道
  [人民日报](https://wap.peopleapp.com/article/rmh36174922/rmh36174922)、[中国网](https://hs.china.com.cn/gd/83980.html)、[光明网](https://health.gmw.cn/2023-06/13/content_36628062.htm)、[TOM科技](https://tech.tom.com/202306/4526869977.html)、[未来网](http://www.zzfuture.cn/news/956.html)、[大众网](http://linyi.dzwww.com.3xw.site/xinwen/202306/t20230613_202306135667.htm)、[中国发展报道网](http://www.chinafzbdw.com/computer/13149.html?1686564408)、[中国日报网](http://energy.chinaduily.com.cn/c/2023/15205.html)、[新华资讯网](http://www.xinhuazxun.com/world/21762.html?1686564382)、[中华网](https://life.china.com/2023-06/12/content_215815.html)、[今日头条](https://www.toutiao.com/article/7243412314223952418/)、[搜狐](https://www.sohu.com/a/684501109_120159010)、[腾讯新闻](https://page.om.qq.com/page/OhSXIMEUtDtdg0rTi6aAoTbg0)、[网易新闻](https://www.163.com/dy/article/I70BJ9U00552UJUX.html)、[中国资讯网](http://www.chinazxun.com/world/23252.html?1686564532)、[中国传播网](http://www.chinachbo.com/a/view/11697.html?1686564509)、[中国都市报道网](http://www.zgdsbdw.com/meida/11273.html?1686564485)、[中华城市网](http://www.zhcsww.com/hot/2023/0612/9609.html?1686564434)

* 公众号
  [广东实验室建设](https://mp.weixin.qq.com/s/gemlKfLg8c-AtjiV7uTUTQ)、[智能语音新青年](https://mp.weixin.qq.com/s/vBMKXUJoAIywkXY2nY60eA)、[深度学习与NLP](https://mp.weixin.qq.com/s/qSHLT8FbvohZESp-UCah6g)、[AINLP](https://mp.weixin.qq.com/s/EX3f9WblLKM8K_nSwhno_g)

## 引用
```bib
@misc{chen2023soulchat,
      title={灵心健康大模型SoulChat：通过长文本咨询指令与多轮共情对话数据集的混合微调，提升大模型的“共情”能力},
      author={Yirong Chen, Xiaofen Xing, Zhenyu Wang, Xiangmin Xu},
      year={2023},
      month = {6},
      version = {1.0},
      url = {https://github.com/scutcyr/SoulChat}
}
```

