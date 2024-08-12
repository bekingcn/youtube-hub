
from html import unescape
import math, os, time
import re
from xml.etree import ElementTree
import requests
from pytubefix import Channel, YouTube, Caption
import streamlit as st
import tiktoken

SUMMARY_SYSTEM_PROMPT = """
- Role: Video Script Summarization Expert
- Background: The user requires a detailed summary of a video script, which involves condensing the key points and themes into a concise format.
- Profile: You are an expert in summarizing video scripts, with a keen eye for detail and the ability to capture the essence of the content.
- Skills: Script analysis, summarization, distillation of key points, concise writing.
- Goals: You will help users to create detailed and accurate summaries of video scripts.
- Constrains: The summary should maintain the original meaning, highlight the main themes, and be concise yet comprehensive.
- OutputFormat: A concise and detailed summary of the video script.
"""

BING_AUTH_URL = "https://edge.microsoft.com/translate/auth"
BING_TRANSLATE_URL = "https://api-edge.cognitive.microsofttranslator.com/translate?from=en&to=zh-CHS&api-version=3.0&includeSentenceLength=true"
BING_TRANSLATE_HEADERS = {
    "accept": "*/*",
    "accept-language": "en,zh-CN;q=0.9,zh;q=0.8,en-GB;q=0.7,en-US;q=0.6",
    "content-type": "application/json",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.77 Safari/537.36 Edg/91.0.864.37",
    "authorization": "",
}


MAX_VEDIOS_PER_CHANNEL = st.secrets.get("MAX_VEDIOS_PER_CHANNEL", 10)
MAX_TOKENS_FOR_SUMMARY = st.secrets.get("MAX_TOKENS_FOR_SUMMARY", 7200)

DATA_FOLDER = "data"
MY_CHANNELS = [c for c in st.secrets.get("MY_CHANNELS", "").split(";") if c.strip()]
DEFAULT_MODEL_LIST = [m for m in st.secrets.get("GROQ_MODEL_LIST", "llama-3.1-8b-instant").split(";") if m.strip()]
DEFAULT_MODEL = "llama-3.1-8b-instant"
DEFAULT_API_KEY = st.secrets.get("GROQ_API_KEY", "")

def summarize_stream(video_texts: str | list, api_key: str, model: str = DEFAULT_MODEL):
    from groq import Groq

    client = Groq(api_key=api_key)
    if isinstance(video_texts, list):
        video_texts = "\n".join(video_texts)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"{video_texts}"
            }
        ],
        temperature=0.3,
        max_tokens=MAX_TOKENS_FOR_SUMMARY,
        top_p=1,
        stream=True,
        stop=None,
    )
    for line in completion:
        yield line.choices[0].delta.content

def summarize(video_texts: str | list, api_key: str, model: str = DEFAULT_MODEL) -> str:
    return " ".join(summarize_stream(video_texts, api_key, model=model))

def translate_stream(text: str | list[str]):
    # refresh token
    resp = requests.get(BING_AUTH_URL)
    auth_token = resp.text

    # translate
    headers = BING_TRANSLATE_HEADERS.copy()
    headers["authorization"] = auth_token
    if isinstance(text, list):
        texts = text
    else:
        texts = text.split("\n")
    # get batchs with 10 lines
    batch_size = 20
    batchs = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    results = []
    for batch in batchs:
        resp = requests.post(BING_TRANSLATE_URL, headers=headers, json=[{"Text": line} for line in batch])
        for line in resp.json():
            yield line["translations"][0]["text"].strip()

def translate(text: str) -> str:
    return "\n".join(translate_stream(text))

def float_to_srt_time_format(d: float, ms: bool = True) -> str:
    """Convert decimal durations into proper srt format.

    :rtype: str
    :returns:
        SubRip Subtitle (str) formatted time duration.

    float_to_srt_time_format(3.89) -> '00:00:03,890'
    """
    fraction, whole = math.modf(d)
    time_fmt = time.strftime("%H:%M:%S", time.gmtime(whole))
    if ms:
        ms = f"{fraction:.3f}".replace("0.", "")
        return f"{time_fmt},{ms}"
    else:
        return time_fmt

def xml_caption_to_time_text(xml_captions: str, merge_lines: int = 1) -> str:
    """Convert xml caption tracks to plain text by line.

    :param str xml_captions:
        XML formatted caption tracks.
    """
    segments = []
    root = ElementTree.fromstring(xml_captions)

    i = 0
    for child in list(root.iter(root.tag))[0]:
        if child.tag == 'p' or child.tag == 'text':
            if "t" in child.attrib:
                start = float(child.attrib["t"]) / 1000.0
            else:
                start = float(child.attrib["start"])
            # unescape the text with html encoding
            caption = unescape(child.text.replace("\n", " ").replace("  ", " "))
            time_text = float_to_srt_time_format(start, False)
            time_text = time_text[3:] if time_text.startswith("00:") else time_text
            segments.append((time_text, caption))
    # merge lines
    total_len = len(segments)
    texts = []
    for i in range(0, total_len, merge_lines):
        time_text, caption = segments[i]
        for j in range(i + 1, min(total_len, i + merge_lines)):
            caption += " " + segments[j][1]
        texts.append((time_text, caption))
    return texts

def merge_time_text_lines(time_text: str, merge_lines: int) -> str:
    lines = time_text.split("\n")
    total_len = len(lines)
    new_lines = []
    for i in range(0, total_len, merge_lines):
        time_len = lines[i].index("]") + 1
        time_text = lines[i][:time_len]
        caption = lines[i][time_len:].strip()
        for j in range(i + 1, min(total_len, i + merge_lines)):
            time_len = lines[j].index("]") + 1
            caption += " " + lines[j][time_len:].strip()
        new_lines.append(f"{time_text} {caption}")
    return new_lines

def to_video_info(video: YouTube):
    return {
        'title': video.title,
        'url': video.watch_url,
        'published': video.publish_date.strftime("%Y-%m-%d %H:%M:%S") if video.publish_date else None,
        'description': video.description,
        'thumbnail': video.thumbnail_url,
        'id': video.video_id,
        "length": float_to_srt_time_format(video.length, ms=False),
        "captions": [caption.code for caption in video.caption_tracks],
    }

def get_xml_caption(video: YouTube, lang: str):
    c = video.captions.get_by_language_code(lang)
    if c:
        xml = c.xml_captions
        return xml, c.xml_caption_to_srt(xml)
    return None, None

def get_channel(channel_id):
    channel = Channel(channel_id)
    videos = channel.videos
    # print(videos)
    info_list = []
    for video in videos[:MAX_VEDIOS_PER_CHANNEL]:
        info_list.append(to_video_info(video))

    return {"videos": info_list, "channel": channel_id}

def search(query):
    from pytubefix import Search
    search = Search(query=query)
    videos = search.videos
    info_list = []
    for video in videos[:MAX_VEDIOS_PER_CHANNEL]:
        info_list.append(to_video_info(video))

    return {"videos": info_list, "query": query}
    

def split_large_text(large_text, max_tokens):
    enc = tiktoken.get_encoding("cl100k_base")
    tokenized_text = enc.encode(large_text)

    chunks = []
    current_chunk = []
    current_length = 0
    total_length = len(tokenized_text)

    for token in tokenized_text:
        current_chunk.append(token)
        current_length += 1

        if current_length >= max_tokens:
            chunks.append(enc.decode(current_chunk).rstrip(' .,;'))
            current_chunk = []
            current_length = 0

    if current_chunk:
        chunks.append(enc.decode(current_chunk).rstrip(' .,;'))

    return chunks[0], total_length

def xset_todo(todo: dict):
    print("enter xset_todo", todo)
    todo = todo or {}
    if "video_id" in todo:
        if not todo["video_id"]:
            st.session_state.todo = {"video_id": st.session_state.video_id}
        else:
            st.session_state.todo = {"video_id": todo["video_id"]}
    elif "channel_id" in todo:
        if not todo["channel_id"]:
            st.session_state.todo = {"channel_id": st.session_state.channel_id}
        elif todo["channel_id"] == "my":
            my_channel = st.session_state.my_channels
            if my_channel:
                st.session_state.todo = {"channel_id": f"https://www.youtube.com/@{my_channel}"}
            else:
                print("No channel_id found. Please set it in the settings")
        else:
            print("Warning: channel_id should be empty or 'my'")
    elif "query" in todo:
        st.session_state.todo = {"query": st.session_state.search_query}
    else:
        st.session_state.todo = {}
    print("", st.session_state.todo)

def get_todo():
    return st.session_state.get("todo", {})

def save_to_json(info, srt=None):
    import json
    folder = DATA_FOLDER
    # create folder if not exists
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # save as json at first
    with open(f"{folder}/{info['id']}.json", "w", encoding="utf-8") as f:
        jstr = json.dumps(info, ensure_ascii=False, indent=4)
        f.write(jstr)
    if srt:
        with open(f"{folder}/{info['id']}.srt", "w", encoding="utf-8") as f:
            f.write(srt)


def preferred_resolution(video: YouTube):
    # we prefer 720p, 480p or 360p
    preferred_resolutions = ["720p", "480p", "360p"]
    defult_stream = video.streams.filter(progressive=True, file_extension="mp4").first()
    stream = video.streams.filter(progressive=True, file_extension="mp4").get_by_resolution(preferred_resolutions[0]) \
        or video.streams.filter(progressive=True, file_extension="mp4").get_by_resolution(preferred_resolutions[1]) \
        or defult_stream
    return stream

def download_video(video_url, pb):
    video = YouTube(video_url)
    stream = preferred_resolution(video)
    title = f"{video.title} ({stream.resolution})"
    pb.progress(0, text=title)
    def _on_progress(stream, chunk, bytes_remaining):
        bytes_downloaded = stream.filesize - bytes_remaining
        percent = bytes_downloaded / stream.filesize
        percent = min(1.0, percent)
        progress = int(100 * percent)
        pb.progress(progress, text=title)
    video.register_on_progress_callback(_on_progress)
    stream.download(output_path=DATA_FOLDER, filename=f"{video.video_id}.mp4", skip_existing=True)
    pb.progress(100, text=title)
    file_path = f"{DATA_FOLDER}/{video.video_id}.mp4"
    srt_path = f"{DATA_FOLDER}/{video.video_id}.srt"
    if not os.path.exists(srt_path):
        srt = video.caption_tracks[0].generate_srt_captions()
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt)
    return file_path, srt_path

def show_video(video: YouTube | None=None, video_id: str=None):
    if video:
        video_id = video.video_id
    local_path = f"{DATA_FOLDER}/{video_id}.mp4"
    srt_local_path = f"{DATA_FOLDER}/{video_id}.srt"
    if not os.path.exists(local_path):
        if video is None:
            video = YouTube(f"https://www.youtube.com/watch?v={video_id}")
        srt = video.caption_tracks[0].generate_srt_captions() if video.caption_tracks else None
        stream = preferred_resolution(video)
        st.video(stream.url, subtitles=srt)
    else:
        st.video(local_path, subtitles=srt_local_path)

def render_existing_video(info):
    # Video info
    st.title(info['title'])
    st.markdown(f"[{info['url']}]({info['url']})")
    st.markdown(f"{info['description'][:100]}...", help=info['description'])
    col1, col2 = st.columns([1, 4])
    download_clicked = col1.button("Fetch to Local", key="fetch_local")
    if download_clicked:
        pb = col2.progress(0, text=info['title'])
        download_video(info['url'], pb)

    
    # Subtitle and translation
    col1, col2 = st.columns([1, 1])
    col1.subheader("Subtitle")
    col2.subheader("Video Preview")

    with col1.container(height=400):
        st.text(info.get('subtitle', 'No subtitle available'))
    with col2.container(height=400):
        show_video(video_id=info['id'])

    col1, col2 = st.columns([1, 1])
    col1.subheader("Translation")
    col2.subheader("Summary")
    with col1.container(height=300):
        st.text("\n\n".join(info.get('translation', ['No translation available'])))
    # Summary
    with col2.container(height=300):
        st.markdown(info.get('summary', 'No summary available'))
    
    
def render_subtitle(video_url, lang):
    # check if the video is already downloaded
    info = st.session_state.get("current_video", {})
    if info and info['url'] == video_url:
        render_existing_video(info)
        return info, False
    
    # Video info
    with st.spinner("Loading video..."):
        video = YouTube(video_url) # , use_oauth=True, allow_oauth_cache=True) # auth for local
        info = to_video_info(video)
        st.session_state["current_video"] = info
        lang = info['captions'][0] if info['captions'] else lang
        print(info)
        st.title(video.title)
        st.markdown(f"[{info['url']}]({info['url']})")
        st.markdown(f"{info['description'][:128]}...", help=info['description'])
    col1, col2 = st.columns([1, 4])
    download_clicked = col1.button("Fetch to Local", key="fetch_local")

    # Subtitle and translation
    col1, col2 = st.columns([1, 1])
    col1.subheader("Subtitle")
    col2.subheader("Video Preview")
    with col1.container(height=400):
        with st.spinner("Captions..."):
            # TODO: no caption available for this video
            xml, srt = get_xml_caption(video, lang)
            texts = xml_caption_to_time_text(xml) if xml else []
            sub_text =  "\n".join([f"[{t[0]}] {t[1]}" for t in texts]).strip()
            st.text(sub_text if sub_text else "No subtitle available")
            info['subtitle'] = sub_text
    with col2.container(height=400):
        show_video(video, video_id=info['id'])
    if not xml:
        return info, True
                

    col1, col2 = st.columns([1, 1])
    col1.subheader("Translation")
    col2.subheader("Summary")
    merge_lines = 2
    with col1.container(height=300):
        with st.spinner("Translating..."):
            tran_list = []
            merged_texts = [
                (texts[i][0], " ".join([t[1] for t in texts[i:i+merge_lines]])) for i in range(0, len(texts), merge_lines)
            ]
            sub_merged =  "\n".join([f"[{t[0]}] {t[1]}" for t in merged_texts]).strip()
            tran_texts = translate_stream(sub_merged) # ["No translation"] # 
            for tran in tran_texts:
                st.text(tran)
                tran_list.append(tran)
            info['translation'] = tran_list
    # Summary
    with col2.container(height=300):
        with st.spinner("Summarizing..."):
            # calculate summary with the maximum of MAX_TOKENS_FOR_SUMMARY tokens
            sub_text, total = split_large_text(sub_text, MAX_TOKENS_FOR_SUMMARY)
            if api_key:
                chunks = []
                def _stream():
                    for t in summarize_stream(sub_text, api_key, model=model):
                        if t:
                            chunks.append(t)
                            yield t
                st.write_stream(_stream)
                info['summary'] = " ".join([c for c in chunks if c])
            else:
                summary = "Please provide an API key to get summary."
                st.markdown(summary)
            if total > MAX_TOKENS_FOR_SUMMARY:
                st.warning(f"Subtitle too long ({total} tokens). Only the first {MAX_TOKENS_FOR_SUMMARY} tokens will be used for summary.")

    return info, srt, True

def render_video_info_list(info_list):
    # display the videos in a table, add a button to download the subtitles
    if info_list:
        title = info_list.get("channel", "") or info_list.get("query", "")
        st.divider()
        st.markdown(f"## Videos [{title}]")
        for info in info_list.get("videos", []):
            with st.container():
                col1, col2 = st.columns([9, 1])
                col1.markdown(f"**[{info['title']}]({info['url']}) - {info['length']} - {info['published']}**")
                col1.markdown(f"{info['description'][:100]}...", help=info['description'])
                col2.button("Subtitles", key=info['id'], on_click=xset_todo, kwargs={"todo": {"video_id": info['url']}})


if __name__ == "__main__":
    # initialize the streamlit settings
    st.set_page_config(page_title="YouTube Subtitle Generator", layout="wide")

    # prepare with todo
    todo = get_todo()
    channel_id = todo.get("channel_id", None)
    video_id = todo.get("video_id", None)
    query = todo.get("query", None)

    # list the videos from a input channel
    st.sidebar.subheader("Channel id")
    st.sidebar.chat_input(key="channel_id", placeholder=channel_id or "Channel id", max_chars=80, on_submit=xset_todo, kwargs={"todo": {"channel_id": ""}})
    st.sidebar.subheader("Video id")
    st.sidebar.chat_input(key="video_id", placeholder=video_id or "Video id", max_chars=80, on_submit=xset_todo, kwargs={"todo": {"video_id": ""}})
    st.sidebar.subheader("Search")
    st.sidebar.chat_input(key="search_query", placeholder=query or "Search ...", max_chars=80, on_submit=xset_todo, kwargs={"todo": {"query": ""}})
    st.sidebar.divider()
    with st.sidebar.expander("Settings", expanded=False):
        lang = st.text_input("Language", "a.en")
        api_key = st.text_input("API key", DEFAULT_API_KEY, type="password")
        model = st.selectbox("Model", DEFAULT_MODEL_LIST, index=0)
    st.sidebar.divider()
    st.sidebar.selectbox("My Channels", MY_CHANNELS, key="my_channels", on_change=xset_todo, kwargs={"todo": {"channel_id": "my"}}, index=None)

    # reset todo to avoid double processing
    xset_todo({})
    info_list = st.session_state.get("info_list", [])
    if video_id:
        info, srt, is_new = render_subtitle(video_id, lang)
        if is_new:
            save_to_json(info, srt)
    elif channel_id:
        with st.spinner("Loading Channel..."):
            info_list = get_channel(channel_id)
        st.session_state["info_list"] = info_list
    elif query:
        with st.spinner("Searching..."):
            info_list = search(query)
        st.session_state["info_list"] = info_list
    
    info = st.session_state.get("current_video", None)
    # show video if nor from channel or search
    if info and not channel_id and not query:
        file_name = info['id'] + "_" + re.sub(r"[^a-zA-Z0-9_]", "_", info["title"].lower().replace(" ", "_"))
        file_name = file_name[:64]
        render_existing_video(info)
        col1, col2, col3, _ = st.columns([1,1,1,2])
        # download the json if it exists
        if os.path.exists(f"{DATA_FOLDER}/{info['id']}.json"):
            col1.download_button("Download JSON", data=open(f"{DATA_FOLDER}/{info['id']}.json", "rb").read(), file_name=f"{file_name}.json")
        if os.path.exists(f"{DATA_FOLDER}/{info['id']}.srt"):
            col2.download_button("Download Srt", data=open(f"{DATA_FOLDER}/{info['id']}.srt", "rb").read(), file_name=f"{file_name}.srt")
        if os.path.exists(f"{DATA_FOLDER}/{info['id']}.mp4"):
            col3.download_button("Download Video", data=open(f"{DATA_FOLDER}/{info['id']}.mp4", "rb").read(), file_name=f"{file_name}.mp4")
    render_video_info_list(info_list)





