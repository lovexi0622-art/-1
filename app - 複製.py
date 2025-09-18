# app.py
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage,TemplateSendMessage,MessageAction,CarouselColumn,CarouselTemplate,TextSendMessage
from openai import OpenAI

app = Flask(__name__)

system_prompt = """
你是一個客服助理，你的名字是小智。
你可以回答以下問題：
1. 產品A保固期多久？ → 2年
2. 產品B保固期多久？ → 1年
3. 如何退換貨？ → 請提供訂單編號，填寫退貨申請
請用禮貌、簡短的方式回答問題、並使用繁體中文。
"""

# 你的 LINE Bot Token 和 Secret
LINE_CHANNEL_ACCESS_TOKEN = "hp3IwT8gL4CBWiQJ+WcVEb1QYl0Vr7tmjMY5nF7SPMncnBsjkSL8lkQvaeCXCP0ObM/jFDBI4QbStd7MRyHlJzgSpJtM0zfUHwmAX1jyKSzK6jqZYkHnGBygCR7wp6xfMjN0iqil1f2q6KslK+DxtQdB04t89/1O/w1cDnyilFU="
LINE_CHANNEL_SECRET = "f5bc15280dc66b140df20b085805d9e9"
OPENAI_API_KEY = "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"

client = OpenAI(api_key=OPENAI_API_KEY)
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

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
    msg = event.message.text
    if msg=="生活相關問題":
        pass
    else:
        resp = client.responses.create(
            model="gpt-4o-mini",        
            input=f"{system_prompt}\n\n使用者問: {msg}\n請回答："
        )
        gpta = resp.output_text
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=gpta)
        )
    
if __name__ == "__main__":
    app.run(port=5000)