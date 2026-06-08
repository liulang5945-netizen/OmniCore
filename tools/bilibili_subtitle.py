"""
B站视频字幕读取插件
自动解析 Bilibili 视频的 CC 字幕，供 Agent 提取和总结
"""
import re
import requests
from langchain_core.tools import Tool

def get_bilibili_subtitles(url: str) -> str:
    """提取B站视频字幕的核心逻辑"""
    try:
        # 1. 使用正则表达式提取 BV 号
        bvid_match = re.search(r'(BV[a-zA-Z0-9]+)', url)
        if not bvid_match:
            return "❌ 无法从链接中提取到有效的 BV 号，请提供包含 BV 号的 B 站视频链接。"
        bvid = bvid_match.group(1)

        # 伪装成浏览器，防止被 B 站拦截
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://www.bilibili.com/"
        }

        # 2. 调用 API 获取视频基本信息 (寻找 cid)
        view_api = f"https://api.bilibili.com/x/web-interface/view?bvid={bvid}"
        res = requests.get(view_api, headers=headers, timeout=10).json()
        if res.get("code") != 0:
            return f"❌ 获取视频信息失败: {res.get('message')}"
        cid = res["data"]["cid"]

        # 3. 调用播放器 API 获取字幕链接列表
        player_api = f"https://api.bilibili.com/x/player/v2?bvid={bvid}&cid={cid}"
        res = requests.get(player_api, headers=headers, timeout=10).json()
        if res.get("code") != 0:
            return f"❌ 获取播放信息失败: {res.get('message')}"

        subtitles = res["data"].get("subtitle", {}).get("subtitles", [])
        if not subtitles:
            return "⚠️ 抱歉，UP 主没有为该视频提供 CC 字幕，无法读取内容。"

        # 4. 获取第一份字幕（通常是中文）并下载
        sub_url = subtitles[0]["subtitle_url"]
        if str(sub_url).startswith("//"):
            sub_url = "https:" + sub_url

        sub_res = requests.get(sub_url, headers=headers, timeout=10).json()
        text_lines = [item["content"] for item in sub_res.get("body", [])]
        full_text = " ".join(text_lines)

        # 5. 限制返回长度，防止爆显存 (通常小模型上下文最大为 4k-8k tokens)
        max_len = 4000
        if len(full_text) > max_len:
            return full_text[:max_len] + "\n\n...(字幕过长已截断)"
        
        return full_text or "⚠️ 字幕内容为空。"

    except Exception as e:
        return f"❌ 解析 B 站字幕时发生内部异常: {str(e)}"

# 暴露给 Agent 的工具列表
TOOLS = [
    Tool(name="read_bilibili_subtitle", description="读取 Bilibili (B站) 视频的官方 CC 字幕内容。输入必须是有效的B站视频完整 URL (例如 https://www.bilibili.com/video/BV...)。", func=get_bilibili_subtitles)
]