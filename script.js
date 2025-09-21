document.getElementById("clickBtn").addEventListener("click", function() {
    const messages = [
        "你真棒！",
        "今天心情不錯吧？",
        "保持微笑 🙂",
        "你學會了新技能！",
        "網頁互動成功！"
    ];
    const randomIndex = Math.floor(Math.random() * messages.length);
    document.getElementById("msg").innerText = messages[randomIndex];
});
