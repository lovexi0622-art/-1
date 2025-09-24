# app.py
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage,TemplateSendMessage,MessageAction,CarouselColumn,CarouselTemplate,TextSendMessage
from openai import OpenAI
import numpy as np
import faiss,os
import re
import time

app = Flask(__name__)

language="繁體中文" #語言設定

# 你的 LINE Bot Token
LINE_CHANNEL_ACCESS_TOKEN = "hp3IwT8gL4CBWiQJ+WcVEb1QYl0Vr7tmjMY5nF7SPMncnBsjkSL8lkQvaeCXCP0ObM/jFDBI4QbStd7MRyHlJzgSpJtM0zfUHwmAX1jyKSzK6jqZYkHnGBygCR7wp6xfMjN0iqil1f2q6KslK+DxtQdB04t89/1O/w1cDnyilFU="
LINE_CHANNEL_SECRET = "f5bc15280dc66b140df20b085805d9e9"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") #GPT Token

client = OpenAI(api_key=OPENAI_API_KEY) #初始化GPT
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN) #初始化Line bot
handler = WebhookHandler(LINE_CHANNEL_SECRET) 


#待定
def chunk_text(text):
    # 只拆換行，不做長度切分，並移除前面的數字編號 (1. 2. 3. ...)
    return [re.sub(r'^\d+\.\s*', '', p.strip()) for p in text.split("\n") if p.strip()]



# --- 讀取本地 FAQ.txt
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
faq_path = os.path.join(BASE_DIR, "FAQ.txt")
with open(faq_path, "r", encoding="utf-8") as f:
    faq_text = f.read()


chunks = []
metadata_list = []
for i, chunk in enumerate(chunk_text(faq_text)):
    chunks.append(chunk)
    metadata_list.append({"doc_id": "FAQ", "chunk_index": i+1})


# --- 2) 叫 embedding API
embeddings = []
batch_size = 16
for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i+batch_size]
    resp = client.embeddings.create(model="text-embedding-3-small", input=batch)
    for item in resp.data:
        embeddings.append(item.embedding)


# --- 3) 建立 FAISS index
emb_array = np.array(embeddings).astype("float32")
faiss.normalize_L2(emb_array)
dim = emb_array.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(emb_array)


# 對照表
id_to_meta = {i: metadata_list[i] for i in range(len(metadata_list))}
id_to_text = {i: chunks[i] for i in range(len(chunks))}

faiss.write_index(index, "knowledge.index")


#使用者問題查詢
def query(q_text, k=5, threshold=0.4):
    # 取得使用者問題的 embedding
    q_emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=q_text
    ).data[0].embedding

    # 轉成 numpy array 並 normalize
    q_arr = np.array([q_emb]).astype("float32")
    faiss.normalize_L2(q_arr)

    # 在 FAISS index 搜尋前 k 個最相似向量
    D, I = index.search(q_arr, k)

    results = []
    for score, idx in zip(D[0], I[0]):
        idx = int(idx)
        # 過濾掉無效索引或相似度低於 threshold 的片段
        if idx == -1 or score < threshold:
            continue

        results.append({
            "text": id_to_text[idx],
            "meta": id_to_meta[idx],
            "score": score
        })

    return results


# LINE webhook 入口
@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return 'OK'


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    global language
    msg = event.message.text #方便後續管理
    if msg == "語言":
        carousel_template = TemplateSendMessage(
            alt_text="這是旋轉木馬選單",
            template=CarouselTemplate(
                columns=[
                    CarouselColumn(
                        title="語言",
                        text="請選擇使用語言",
                        actions=[
                            MessageAction(label="中文", text="繁體中文"),
                            MessageAction(label="英文", text="英文")
                        ]
                    )
                ]
            )
        )
        line_bot_api.reply_message(event.reply_token, carousel_template)

    elif msg == "繁體中文":
        language="繁體中文"

    elif msg == "英文":
        language="英文"

    else:

        try:

            line_bot_api.push_message(event.reply_token, TextSendMessage(text="GPT正在思考中..."))#如果line不擋就輸出"GPT正在思考中..."

        except Exception:
            pass

        #只取5個相似度大於 0.4 的片段
        retrieved = query(msg, k=5, threshold=0.4)  

        #有FAQ資訊
        if retrieved:  
            context = "\n\n---\n\n".join(
                f"[來源: {r['meta']['chunk_index']}] {r['text']} [相似度: {r['score']}]" for r in retrieved
            )
            prompt = f"""你是一個元培機器人。請基於以下引用的文件片段回答使用者問題，若與學校相關且無相關資料請說「找不到相關資訊」。
            引用:
            {context}

            使用者問題: {msg}
            請用{language}簡短回答"""
            print("回應:",context) #後台確認傳給GPT資料

        #無FAQ資訊 GPT自己判斷怎麼回
        else:  
            prompt = f"""你是一個元培機器人。請回答使用者問題，但如果與學校相關且無資料，請說「找不到相關資訊」。
            使用者問題: {msg}
            請用{language}簡短回答。"""


        gpt5_models = ["gpt-5-nano","gpt-4o-mini","gpt-5-mini"]  #GPT模型
        resp = None

        #依序執行GPT模型
        for model_name in gpt5_models:
            try:
                resp = client.responses.create(model=model_name, input=prompt)
                print("GPT型號:",model_name)
                break
            except Exception as e:
                print(f"{model_name} 呼叫失敗: {e}")
                time.sleep(0.1)
                continue  #嘗試下一個模型


        if resp is None:
            # 所有模型都失敗，回傳預設訊息
            print("目前無法取得回應，請稍後再試。")
        else:
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=resp.output_text)) # GPT回復
    

if __name__ == "__main__":
    app.run(port=5000)
