from langchain_community.vectorstores import Pinecone as LcPc
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from pinecone import Pinecone
import streamlit as st
import time
import re
import hmac



# Authentication:
def check_password():
    """Returns `True` if the user had a correct password."""

    def login_form():
        """Form with widgets to collect user information"""
        with st.form("Credentials"):
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.form_submit_button("Log in", on_click=password_entered)



    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["username"] in st.secrets[
            "passwords"
        ] and hmac.compare_digest(
            st.session_state["password"],
            st.secrets.passwords[st.session_state["username"]],
        ):
            st.session_state["password_correct"] = True

            # Don't store the username or password
            del st.session_state["password"]
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False


    # Return True if the username + password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show inputs for username + password.
    login_form()
    if "password_correct" in st.session_state:
        st.error("Username or password incorrect")
    return False

# Do not continue until authenticated
if not check_password():
    st.stop()



# BEGIN WEBPAGE:
OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']
PINECONE_API_KEY = st.secrets['PINECONE_API_KEY']
PINECONE_API_ENV = st.secrets['ENVIRONMENT']
index_name = st.secrets['INDEX_NAME']
page_title = st.secrets['PAGE_TITLE']


# Create an OpenAI embeddings instance
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Initialize Pinecone docsearch
pc = Pinecone(api_key=PINECONE_API_KEY)
# index = pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
docsearch = LcPc.from_existing_index(index_name, embeddings)

# Create an OpenAI LLM (Language Model) instance
llm = ChatOpenAI(temperature=0.3, openai_api_key=OPENAI_API_KEY, model='gpt-4')
chain = load_qa_chain(llm, chain_type="stuff")


# Create Question and Answer function
def QnA(question):
    """Returns the answer and source documents to the input question"""
    docs = docsearch.similarity_search(question)
    answer = chain.run(input_documents=docs, question=question)
    return answer, docs


# Title
st.title(page_title)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=message["avatar"]):
        st.markdown(message["content"])

user_image = '🕵️'
assistant_image = '🤖'

# Accept user input
if prompt := st.chat_input("Ask me anything"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "avatar": user_image, "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user", avatar=user_image):
        st.markdown(prompt)

    # Retrieve a response
    response = QnA(prompt)
    answer = response[0]
    docs = response[1]

    # Create list of filenames
    files = []
    for document in docs:
        filename = document.metadata["filename"]
        if filename not in files:
            files.append(filename)

    # Display assistant response in chat message container
    with st.chat_message("assistant", avatar=assistant_image):
        message_placeholder = st.empty()
        filenames_placeholder = st.empty()
        full_response = ""

        assistant_response = answer + '  \n\n **Files:**  \n' + '  \n'.join(files)

        # Simulate stream of response with milliseconds delay
        for chunk in re.split(r"(\s+)", assistant_response):
            full_response += chunk + " "
            time.sleep(0.02)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "avatar": assistant_image, "content": full_response})
