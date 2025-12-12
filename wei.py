import streamlit as st
st.set_page_config(page_title="音乐",page_icon="🎵")
st.title("🎵简易音乐播放器")

if 'song_ind' not in st.session_state:
    st.session_state['song_ind'] = 0

# 歌曲列表（包含封面、标题、歌手、时长、音频链接）
songs = [
    {
        "cover": "http://p2.music.126.net/lHrmzaGOd-PmrlmVxJfWyg==/109951172350451653.jpg?param=130y130",# 音频封面
        "title": "海屿你",
        "singer": "小奥奥",
        "duration": "4:56",
        "audio_url": "https://music.163.com/song/media/outer/url?id=3322383475.mp3"  # 音频链接
    },
    {
        "cover": "http://p2.music.126.net/YiYv2SfEno67XeuEVwDhXQ==/109951169849771260.jpg?param=130y130",
        "title": "Sneaky Tactics",
        "singer": "Ilona Harpaz",
        "duration": "2:31",
        "audio_url": "https://music.163.com/song/media/outer/url?id=2614865529.mp3"  
    },
    {
        "cover": "http://p2.music.126.net/EDhgL1S2DLGVE_5cjU-hfQ==/109951172410328709.jpg?param=130y130",
        "title": "大东北是我的家乡",
        "singer": "袁娅维TIA RAY",
        "duration": "3:46",
        "audio_url": "https://music.163.com/song/media/outer/url?id=3327141886.mp3"
    }
]

# 获取当前歌曲信息
current_song = songs[st.session_state['song_ind']]

# 布局：封面+歌曲信息
col_cover, col_info = st.columns([1, 2])
with col_cover:
    st.image(current_song["cover"], caption="专辑封面", width=150)
with col_info:
    st.subheader(current_song["title"])
    st.write(f"歌手: {current_song['singer']}")
    st.write(f"时长: {current_song['duration']}")

# 核心：播放当前歌曲的音频
st.subheader("🎧 播放音频")
st.audio(current_song["audio_url"], format="audio/mp3", start_time=0)

# 切换歌曲函数
def next_song():
    st.session_state['song_ind'] = (st.session_state['song_ind'] + 1) % len(songs)

def prev_song():
    st.session_state['song_ind'] = (st.session_state['song_ind'] - 1) % len(songs)

c1, c2 = st.columns([2, 2])
with c1:
    st.button("|◀|上一首", on_click =prev_song, use_container_width=True)
with c2:
    st.button("|▶|下一首", on_click =next_song, use_container_width=True)



