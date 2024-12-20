# 基于检索增强问答的$LLM$调用示例及服务接口

-----

本项目针对专有资料库，通过检索返回相关文档，并基于检索结果优化提示词，调用$LLM$生成答案（包含智谱AI与讯飞星火两种模型）。

## 环境准备
-----
### 1. 安装依赖
```shell
pip install -r requirements.txt
```

### 2. 配置环境变量
项目根目录下新建`.env`文件，根据需求添加以下内容：
```shell
spark_appid=""
spark_api_key=""
spark_api_secret=""

ZHIPUAI_API_KEY=""

HOST = ""
PORT = 
```

## 开始使用
-----
### 1. 向量库构建
```
python db/create_db.py
```
需要设定`DATA_FOLDER_PATH`与`PERSIST_DIRECTORY`，分别表示资料库路径与向量库保存路径。

### 2. 可视化界面
```shell
streamlit run interface/app.py
```

### 3. 服务接口
```shell
python app/main.py
```
获取服务时需要对用户信息进行校验，可按需参考`app/key.py`中的`KeyChecker`类修改。
调用返回：
```json
{
    "question":"xxx",
    "response":"xxx"
}
```

## 注意事项
-----
- 模型调用接口为第三方提供，请自行申请相关接口权限。
- 所用资料库为个人收集，请勿用于商业用途。