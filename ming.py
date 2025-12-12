import streamlit as st

st.set_page_config(page_title="相册",page_icon="😃")
st.title("我的相册")

if 'ind' not in st.session_state:
    st.session_state['ind']=0

images = [
    {
        'url': "https://www.baltana.com/files/wallpapers-2/Cute-Cat-Images-07756.jpg",
        'text': '猫'
    },
    {
        'url': "https://cdn.britannica.com/82/232782-050-8062ACFA/Black-labrador-retriever-dog.jpg",
        'text': 'dog'
    },
    {
        'url': "https://live.staticflickr.com/2686/4497672316_d283310530_3k.jpg",
        'text': 'lion'
    }
]

# url:图片的地址  caption:图片注释介绍
st.image(images[st.session_state['ind']]['url'], caption=images[st.session_state['ind']]['text'])


def nextImg():
    st.session_state['ind'] = (st.session_state['ind'] + 1) % len(images)


# 分列容器
cl, c2 = st.columns(2)

with cl:
    st.button("上一张", on_click=nextImg, use_container_width=True)

with c2:
    # 按钮
    st.button("下一张", on_click=nextImg, use_container_width=True)
