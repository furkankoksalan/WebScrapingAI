import streamlit as st
import requests
from bs4 import BeautifulSoup
import json
import os
from datetime import datetime
import uuid
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import WebBaseLoader
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler

load_dotenv()

st.set_page_config(
    page_title="Web Scraping + AI Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: flex-start;
    }
    .chat-message.user {
        background-color: #2b313e;
        flex-direction: row-reverse;
    }
    .chat-message.bot {
        background-color: #475063;
    }
    .chat-message .message {
        flex: 1;
        padding: 0 1rem;
    }
</style>
""", unsafe_allow_html=True)


class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""
        self.placeholder = None

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        if self.placeholder is None:
            self.placeholder = self.container.empty()
        self.placeholder.markdown(self.text)


class WebScrapingChatbot:
    def __init__(self):
        self.data_dir = "chat_data"
        self.sessions_file = os.path.join(self.data_dir, "sessions.json")
        self.scraped_data_dir = os.path.join(self.data_dir, "scraped_data")
        self.setup_directories()

    def setup_directories(self):
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.scraped_data_dir, exist_ok=True)

        if not os.path.exists(self.sessions_file):
            with open(self.sessions_file, 'w', encoding='utf-8') as f:
                json.dump({}, f)

    def scrape_website(self, url: str) -> Dict[str, Any]:
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()

            title = soup.find('title')
            title_text = title.get_text().strip() if title else "Başlık bulunamadı"

            content_selectors = ['article', 'main', '.content', '#content', '.post', '.entry']
            content = ""

            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    content = ' '.join([elem.get_text().strip() for elem in elements])
                    break

            if not content:
                content = soup.get_text()

            lines = [line.strip() for line in content.split('\n') if line.strip()]
            clean_content = ' '.join(lines)

            return {
                'url': url,
                'title': title_text,
                'content': clean_content[:5000],
                'scraped_at': datetime.now().isoformat(),
                'success': True
            }

        except Exception as e:
            return {
                'url': url,
                'error': str(e),
                'scraped_at': datetime.now().isoformat(),
                'success': False
            }

    def create_vector_store(self, scraped_data: List[Dict[str, Any]], openai_api_key: str):
        try:
            from langchain.docstore.document import Document

            documents = []
            for data in scraped_data:
                if data.get('success') and data.get('content'):
                    doc = Document(
                        page_content=data['content'],
                        metadata={
                            'url': data['url'],
                            'title': data.get('title', ''),
                            'scraped_at': data['scraped_at']
                        }
                    )
                    documents.append(doc)

            if not documents:
                return None

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)

            embeddings = OpenAIEmbeddings(api_key=openai_api_key)
            vectorstore = FAISS.from_documents(splits, embeddings)

            return vectorstore

        except Exception as e:
            st.error(f"Vektör deposu oluşturulurken hata: {str(e)}")
            return None

    def get_sessions(self) -> Dict:
        try:
            with open(self.sessions_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}

    def save_sessions(self, sessions: Dict):
        with open(self.sessions_file, 'w', encoding='utf-8') as f:
            json.dump(sessions, f, ensure_ascii=False, indent=2)

    def create_new_session(self) -> str:
        session_id = str(uuid.uuid4())
        sessions = self.get_sessions()
        sessions[session_id] = {
            'created_at': datetime.now().isoformat(),
            'title': f"Sohbet {len(sessions) + 1}",
            'messages': [],
            'scraped_urls': []
        }
        self.save_sessions(sessions)
        return session_id


def display_chat_message(role: str, content: str):
    """Chat mesajlarını göster"""
    if role == "user":
        with st.chat_message("user"):
            st.markdown(content)
    else:
        with st.chat_message("assistant"):
            st.markdown(content)


def main():
    st.title("Web Scraping + AI Chat Uygulaması")
    st.markdown("Web sitelerinden bilgi çekerek AI ile sohbet edin")

    if not os.getenv("OPENAI_API_KEY"):
        st.error("OpenAI API anahtarı gerekli")
        st.markdown("""
        ### Kurulum:

        1. `.env` dosyası oluşturun
        2. API anahtarınızı ekleyin:
           ```
           OPENAI_API_KEY=sk-your-openai-api-key-here
           ```
        3. Uygulamayı yeniden başlatın

        API anahtarı: [OpenAI Platform](https://platform.openai.com/api-keys)
        """)
        st.stop()

    st.success("API anahtarı yüklendi")

    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = WebScrapingChatbot()

    with st.sidebar:
        st.header("Ayarlar")

        openai_api_key = os.getenv("OPENAI_API_KEY")

        st.markdown("### Model Seçimi")
        gpt_models = {
            "GPT-4o Mini": "gpt-4o-mini",
            "GPT-3.5 Turbo": "gpt-3.5-turbo",
            "GPT-4": "gpt-4",
            "GPT-4 Turbo": "gpt-4-turbo-preview",
            "GPT-4o": "gpt-4o"
        }

        selected_model_name = st.selectbox(
            "GPT Modeli",
            list(gpt_models.keys()),
            help="Kullanmak istediğiniz GPT modelini seçin"
        )
        selected_model = gpt_models[selected_model_name]

        model_info = {
            "gpt-4o-mini": "Hızlı ve ekonomik",
            "gpt-3.5-turbo": "Hızlı ve etkili",
            "gpt-4": "Yüksek kalite",
            "gpt-4-turbo-preview": "Gelişmiş GPT-4",
            "gpt-4o": "En yeni model"
        }

        if selected_model in model_info:
            st.info(model_info[selected_model])

        with st.expander("Gelişmiş Ayarlar"):
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1
            )

            max_tokens = st.slider(
                "Maksimum Token",
                min_value=100,
                max_value=4000,
                value=2000,
                step=100
            )

            streaming = st.checkbox("Streaming Mod", value=True, help="Cevapları gerçek zamanlı göster")

        st.markdown("### Sohbet Oturumları")

        sessions = st.session_state.chatbot.get_sessions()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Yeni Sohbet", use_container_width=True):
                new_session_id = st.session_state.chatbot.create_new_session()
                st.session_state.current_session = new_session_id
                st.rerun()

        with col2:
            if st.button("Tümünü Sil", use_container_width=True):
                if st.session_state.get('confirm_delete', False):
                    st.session_state.chatbot.save_sessions({})
                    if 'current_session' in st.session_state:
                        del st.session_state.current_session
                    st.session_state.confirm_delete = False
                    st.rerun()
                else:
                    st.session_state.confirm_delete = True
                    st.warning("Tekrar tıklayın")

        if sessions:
            st.markdown("**Mevcut Sohbetler:**")
            for session_id, session_data in sessions.items():
                if st.button(f"{session_data['title']}",
                             key=f"session_{session_id}",
                             use_container_width=True):
                    st.session_state.current_session = session_id
                    st.rerun()

        st.markdown("### Web Scraping")
        urls_input = st.text_area(
            "URL'ler (her satıra bir tane)",
            placeholder="https://example.com\nhttps://example2.com",
            height=100
        )

        if st.button("Web Sitelerini Tara", use_container_width=True):
            if urls_input.strip():
                urls = [url.strip() for url in urls_input.split('\n') if url.strip()]

                if urls:
                    with st.spinner("Web siteleri taranıyor..."):
                        scraped_data = []
                        progress_bar = st.progress(0)

                        for i, url in enumerate(urls):
                            st.text(f"Taranıyor: {url}")
                            data = st.session_state.chatbot.scrape_website(url)
                            scraped_data.append(data)
                            progress_bar.progress((i + 1) / len(urls))

                        if 'current_session' not in st.session_state:
                            st.session_state.current_session = st.session_state.chatbot.create_new_session()

                        sessions = st.session_state.chatbot.get_sessions()
                        current_session = sessions[st.session_state.current_session]
                        current_session['scraped_urls'].extend(
                            [data['url'] for data in scraped_data if data['success']])
                        st.session_state.chatbot.save_sessions(sessions)

                        vectorstore = st.session_state.chatbot.create_vector_store(scraped_data, openai_api_key)
                        if vectorstore:
                            st.session_state.vectorstore = vectorstore
                            st.success(f"{len([d for d in scraped_data if d['success']])} web sitesi tarandı")
                        else:
                            st.error("Vektör deposu oluşturulamadı")

                        for data in scraped_data:
                            if data['success']:
                                st.success(f"✅ {data['url']}")
                            else:
                                st.error(f"❌ {data['url']}: {data['error']}")
            else:
                st.warning("Lütfen en az bir URL girin")

    # Oturum yönetimi
    if 'current_session' not in st.session_state:
        if sessions:
            st.session_state.current_session = list(sessions.keys())[0]
        else:
            st.session_state.current_session = st.session_state.chatbot.create_new_session()

    sessions = st.session_state.chatbot.get_sessions()
    if st.session_state.current_session in sessions:
        current_session = sessions[st.session_state.current_session]

        st.subheader(f"{current_session['title']}")

        if current_session.get('scraped_urls'):
            with st.expander("Taranan Web Siteleri"):
                for url in current_session['scraped_urls']:
                    st.write(f"• {url}")

        # Chat geçmişini göster
        for message in current_session['messages']:
            display_chat_message(message['role'], message['content'])

        # Chat input
        if prompt := st.chat_input("Mesajınızı yazın..."):
            # Kullanıcı mesajını ekle ve göster
            current_session['messages'].append({
                'role': 'user',
                'content': prompt,
                'timestamp': datetime.now().isoformat()
            })

            display_chat_message("user", prompt)

            try:
                # AI cevabı için container oluştur
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()

                    # LLM oluştur
                    llm = ChatOpenAI(
                        model=selected_model,
                        api_key=openai_api_key,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        streaming=streaming
                    )

                    if hasattr(st.session_state, 'vectorstore') and st.session_state.vectorstore:
                        # RAG ile cevap
                        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})

                        chat_history = []
                        for msg in current_session['messages'][:-1]:
                            if msg['role'] == 'user':
                                chat_history.append(HumanMessage(content=msg['content']))
                            else:
                                chat_history.append(AIMessage(content=msg['content']))

                        if streaming:
                            # Streaming callback handler
                            callback_handler = StreamlitCallbackHandler(message_placeholder)

                            qa_chain = ConversationalRetrievalChain.from_llm(
                                llm=llm,
                                retriever=retriever,
                                return_source_documents=True,
                                verbose=False
                            )

                            result = qa_chain({
                                "question": prompt,
                                "chat_history": chat_history
                            }, callbacks=[callback_handler])
                        else:
                            # Normal mode
                            qa_chain = ConversationalRetrievalChain.from_llm(
                                llm=llm,
                                retriever=retriever,
                                return_source_documents=True,
                                verbose=False
                            )

                            result = qa_chain({
                                "question": prompt,
                                "chat_history": chat_history
                            })

                            message_placeholder.markdown(result['answer'])

                        ai_response = result['answer']

                        # Kaynakları ekle
                        if result.get('source_documents'):
                            sources = set([doc.metadata.get('url', 'Bilinmeyen kaynak')
                                           for doc in result['source_documents']])
                            sources_text = f"\n\n**Kaynaklar:**\n" + "\n".join([f"• {source}" for source in sources])
                            ai_response += sources_text
                            message_placeholder.markdown(ai_response)

                    else:
                        # Normal chat
                        system_prompt = """Sen yardımksever bir AI asistanısın. Türkçe olarak net ve faydalı cevaplar ver. 
                        Web scraping ile elde edilen bilgiler varsa onları kullan, yoksa genel bilgilerinle yardım et."""

                        if streaming:
                            # Streaming response
                            full_response = ""
                            for chunk in llm.stream([HumanMessage(content=f"{system_prompt}\n\nKullanıcı: {prompt}")]):
                                full_response += chunk.content
                                message_placeholder.markdown(full_response + "▌")
                            message_placeholder.markdown(full_response)
                            ai_response = full_response
                        else:
                            # Normal response
                            response = llm.invoke([HumanMessage(content=f"{system_prompt}\n\nKullanıcı: {prompt}")])
                            ai_response = response.content
                            message_placeholder.markdown(ai_response)

                # Cevabı kaydet
                current_session['messages'].append({
                    'role': 'assistant',
                    'content': ai_response,
                    'timestamp': datetime.now().isoformat()
                })

                sessions[st.session_state.current_session] = current_session
                st.session_state.chatbot.save_sessions(sessions)

            except Exception as e:
                st.error(f"Hata oluştu: {str(e)}")
                # Hatalı mesajı sil
                if current_session['messages'] and current_session['messages'][-1]['role'] == 'user':
                    current_session['messages'].pop()

    else:
        st.error("Sohbet oturumu bulunamadı")
        if st.button("Yeni Sohbet Başlat"):
            st.session_state.current_session = st.session_state.chatbot.create_new_session()
            st.rerun()


if __name__ == "__main__":
    main()