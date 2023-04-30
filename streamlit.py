import os
import openai
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter# from langchain.text_splitter import split_text
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.docstore.document import Document
from dotenv import load_dotenv
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter


load_dotenv() # Load variables from .env file

def generate_questions(text):
    # Construct the prompt for the GPT model
    prompt = f"Generate 1 true false or factual very short answer test question based on this text, and do not include the answer: {text}"  # noqa: E501

    # Call the OpenAI API for chat completion
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a teacher. I need you to help write me exam questions.",  # noqa: E501
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=1200,
        n=1,
        stop=None,
        temperature=0.7,
    )

    # Extract the generated HTML from the response
    reply = response["choices"][0]["message"]["content"]
    return reply

def evaluate_questions(query, result, user_answer):
    # Construct the prompt for the GPT model
    prompt = f"give encouraging feedback to this student as if you were quizzing them on this question, telling them if they are correct and how to improve if applicable: {query} 'correct answer': ' {result['result']}', student answer: {user_answer}"  # noqa: E501

    # Call the OpenAI API for chat completion
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a teacher. I need you to help grade exams.",  # noqa: E501
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=1200,
        n=1,
        stop=None,
        temperature=0.7,
    )

    # Extract the generated HTML from the response
    reply = response["choices"][0]["message"]["content"]
    return reply


class Message:
    ai_icon = "./img/robot.png"

    def __init__(self, label: str, expanded: bool = True):
        self.label = label
        self.expanded = expanded

    def __enter__(self):
        message_area, icon_area = st.columns([10, 1])
        icon_area.image(self.ai_icon, caption="QuizMeGPT")

        self.expander = message_area.expander(label=self.label, expanded=self.expanded)

        return self

    def __exit__(self, ex_type, ex_value, trace):
        pass

    def write(self, content):
        self.expander.markdown(content)

# def split_text(text):
#     # """Split documents."""
#     texts = text
#     # metadatas = [doc.metadata for doc in documents]
#     documents = []
#     for i, text in enumerate(texts):
#         for chunk in split_text(text):
#             new_doc = Document(
#                 page_content=chunk
#             )
#             documents.append(new_doc)
#     return documents




def quizflow(text, openai_api_key):
    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 1000,
        chunk_overlap  = 0,
        # length_function = len,
    )
    # texts = text_splitter.split_text(text)
    texts = text_splitter.create_documents([text])

    # Select which embeddings we want to use
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Create the vector store to use as the index
    db = Chroma.from_documents(texts, embeddings)

    # Expose this index in a retriever interface
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    # Create a chain to answer questions
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever, return_source_documents=True)
    # Generate a question
    query = generate_questions(text)

    with Message(label="Question") as m:
        m.write("### Question")
        m.write(query)

    user_answer = st.text_input("Write your answer here:")
    result = qa({"query": query})

    submit_button = st.button("Submit")

    if submit_button:
        feedback = evaluate_questions(query, result, user_answer)
        provide_feedback(feedback, text, openai_api_key)

def provide_feedback(feedback, text, openai_api_key):
    with Message(label="Feedback") as m:
        m.write("### Here's how you did!")
        m.write(feedback)

    new_question_button = st.button("New Question")
    if new_question_button:
        quizflow(text, openai_api_key)

if __name__ == "__main__":
    st.set_page_config(
        initial_sidebar_state="expanded",
        page_title="QuizMeGPT Streamlit",
        layout="centered",
    )

    with st.sidebar:
        openai_api_key = st.text_input('Your OpenAI API KEY', type="password")

    st.title("QuizMeGPT")
    text = st.text_area("Paste your text here", height=300)
    if text:
        quizflow(text, openai_api_key)
