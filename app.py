from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage,TemplateSendMessage,MessageAction,CarouselColumn,CarouselTemplate,TextSendMessage,FlexSendMessage
from openai import OpenAI
import numpy as np
import faiss,os,re,time,random

app = Flask(__name__)

language="繁體中文" #語言設定
gpt5_models = ["gpt-5-nano","gpt-4o-mini","gpt-5-mini"]  #GPT模型
flex_color = [["#F9FAFB","#2563EB","#1F2937"],["#FFF8F0","#E07A5F","#374151"],["#ECFDF5","#10B981","#064E3B"]] # Flex顏色設定

# 你的 LINE Bot Token
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") #GPT API

client = OpenAI(api_key=OPENAI_API_KEY) #初始化GPT
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN) #初始化Line bot
handler = WebhookHandler(LINE_CHANNEL_SECRET) #Line驗證金鑰


#將FAQ.TXT格式化，以\n為斷點並去編號
def chunk_text(text):
    '''
    將FAQ每行分好段落text.split("\n")。並用if p.strip()防呆，跳過空白行
    將各行進一步加工^\d+\.\s* 刪除每行開頭(^)的多個數字(\d+)及小數點(\.)
    '''
    return [re.sub(r'^\d+\.', '', p) for p in text.split("\n") if p.strip()]


# --- 讀取本地 FAQ.txt
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
faq_path = os.path.join(BASE_DIR, "FAQ.txt")
with open(faq_path, "r", encoding="utf-8") as f:
    faq_text = f.read()


# --- 回應分段處理
chunks = []
metadata_list = []
for i, chunk in enumerate(chunk_text(faq_text)):
    chunks.append(chunk)
    metadata_list.append({"doc_id": "FAQ", "chunk_index": i+1})


# ---  叫 embedding API
embeddings = []
batch_size = 16
for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i+batch_size]
    resp = client.embeddings.create(model="text-embedding-3-small", input=batch)
    for item in resp.data:
        embeddings.append(item.embedding)


# ---  建立 FAISS index
emb_array = np.array(embeddings).astype("float32")
faiss.normalize_L2(emb_array)
dim = emb_array.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(emb_array)


# 對照表
id_to_meta = {i: metadata_list[i] for i in range(len(metadata_list))} #來源
id_to_text = {i: chunks[i] for i in range(len(chunks))} #內容


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

# 以上初始化程式


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


# 確認訊息是否是電話相關訊息
def phone(text):
    ph1=[]
    ph1.append(re.search(r"03-\d{7}", text).group())
    start = text.find(ph1[0])
    left = text.rfind("•", 0, start)+1 # 往前找最近的分隔符 •
    ph1.append(text[left:start-1])
    return ph1


# 訊息事件處理
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):

    #方便後續整理
    global language #沒什麼用 全域變數宣告
    timen={} #程式運行時間軸 debug用的
    start_time = time.time() #紀錄起始時間 debug用的
    msg = event.message.text #方便後續管理


    if msg:

        #只取5個相似度大於 0.4 的片段
        retrieved = query(msg, k=5, threshold=0.4)  
        timen["問題比對"]= time.time() - start_time #時間紀錄 debug用的

        # 確認給GPT的回應的格式
        if retrieved: #有FAQ資訊

            context = "\n\n---\n\n".join(f"{r['text']}" for r in retrieved)

            prompt = f"""你是一個元培機器人、代表元培醫事科技大學。請基於以下引用的文件片段回答使用者問題，若與學校相關且無相關資料請說「找不到相關資訊」。
            引用:
            {context}

            使用者問題: {msg}
            請用{language}以及最精準且簡短並以最快的速度回復。使用「•」作為前綴"""

            
            test = "\n---\n".join(f"[來源: {r['meta']['chunk_index']}] {r['text']} [相似度: {r['score']}]" for r in retrieved)
            print("回應:",test) #後台確認傳給GPT資料 debug用的

        else: #無FAQ資訊 GPT自己判斷怎麼回
            prompt = f"""你是一個元培機器人、代表元培醫事科技大學。請回答使用者問題，但如果與學校相關，請先引導讓使用知道怎麼問更加精準。若無法引導請回「找不到相關資訊」。
            使用者問題: {msg}
            請用{language}以及最精準且簡短並以最快的速度回復。使用「•」作為前綴"""


        timen["回答方式選擇"]= time.time() - start_time #時間紀錄
        resp = None

        #依序執行GPT模型
        for model_name in gpt5_models:
            try:
                resp = client.responses.create(model=model_name, input=prompt) # 呼叫GPT
                print("GPT型號:",model_name)
                break
            except Exception as e:
                print(f"{model_name} 呼叫失敗: {e}")
                time.sleep(0.1)
                continue  #嘗試下一個模型


        timen["GPT回答"]= time.time() - start_time
        if resp is None:

            # 所有模型都失敗，回傳預設訊息
            print("GPT全故障，請檢查是否TPD OR API出問題")
            line_bot_api.reply_message(event.reply_token, "目前無法取得回應，請稍後再試。")

        else:

            rdcolor=random.randint(0,2) # 隨機色系

            if resp.output_text.count("03-")==1 and "電話" in resp.output_text: # 確認是否需要出現電話的按鈕

                ph=phone(resp.output_text)
                flex_message = FlexSendMessage(
                    alt_text="元培資管機器人",
                    contents={
                        "type": "bubble",
                        "body": {
                            "type": "box",
                            "layout": "vertical",
                            "backgroundColor": flex_color[rdcolor][0],
                            "contents": [
                                {
                                    "type": "text",
                                    "text": "元培資管機器人",
                                    "weight": "bold",
                                    "size": "xl",
                                    "align": "center",
                                    "color": flex_color[rdcolor][1]
                                },
                                {
                                        "type": "separator",
                                        "margin": "md",
                                },
                                {
                                    "type": "text",
                                    "text": resp.output_text,
                                    "wrap": True,
                                    "size": "sm",
                                    "align": "start",
                                    "color": flex_color[rdcolor][2],
                                    "margin": "md"
                                }
                            ]
                        },"footer": {
                            "type": "box",
                            "layout": "vertical",
                            "spacing": "sm",
                            "backgroundColor": flex_color[rdcolor][0],
                            "contents": [
                                {
                                    "type": "button",
                                    "style": "link",
                                    "action": {
                                        "type": "uri",
                                        "label": f"📞 聯絡{ph[1]}",
                                        "uri": f"tel://{ph[0]}"
                                    }
                                }
                            ]
                        }
                    }
                )

            else:

                flex_message = FlexSendMessage(
                    alt_text="元培資管機器人",
                    contents={
                        "type": "bubble",
                        "body": {
                                    "type": "box",
                                    "layout": "vertical",
                                    "backgroundColor":flex_color[rdcolor][0] ,
                                    "contents": [
                                {
                                    "type": "text",
                                    "text": "元培資管機器人",
                                    "weight": "bold",
                                    "size": "xl",
                                    "align": "center",
                                    "color": flex_color[rdcolor][1]  
                                },
                                {
                                    "type": "separator",
                                    "margin": "md",
                                    "color": "#000000"
                                },
                                {
                                    "type": "text",
                                    "text": resp.output_text,
                                    "wrap": True,
                                    "size": "sm",
                                    "align": "start",
                                    "color": flex_color[rdcolor][2],
                                    "margin": "md"
                                }
                            ]
                        }
                    }
                )

            line_bot_api.reply_message(event.reply_token, flex_message) # 將訊息回傳

            #後台運行速度紀錄
            timen["LINE機器人回復"]= time.time() - start_time
            for i in timen.keys():
                print(i,":",timen[i],end=" ")
            print("")

# 讓程式在通訊埠5000運行
if __name__ == "__main__":
    app.run(port=5000)
