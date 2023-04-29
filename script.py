import os
import openai
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from dotenv import load_dotenv

load_dotenv() # Load variables from .env file


def generate_questions(doc):
    # Construct the prompt for the GPT model
    prompt = f"Generate 1 true false or factual very short answer test question based on this text, and do not include the answer: {doc}"  # noqa: E501

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


if __name__ == "__main__":
    # load document
    api_key = os.environ.get("OPENAI_API_KEY") 
    loader = PyPDFLoader("./sharks.pdf")
    documents = loader.load()
    
    # split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    # select which embeddings we want to use
    embeddings = OpenAIEmbeddings(openai_api_key = api_key)

    # create the vectorestore to use as the index
    db = Chroma.from_documents(texts, embeddings)

    # expose this index in a retriever interface
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":2})

    # create a chain to answer questions 
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever, return_source_documents=True)
    # print("TEXT", texts[5])
    query = generate_questions(texts[5])
    print ('********************\n\n')
    print('GENERATED QUESTION:', query, '\n\n')
    result = qa({"query": query})
    print ('********************\n')
    user_answer = input("Enter your answer: ")
    print ('\n********* Generating Feedback ***********\n')
    feedback = evaluate_questions(query, result, user_answer)
    print('FEEDBACK:', feedback)

    print ('\n\n\n********* Complete Answer ***********\n')  
    print('CORRECT ANSWER:', result['result'])
    print ('\n\n\n********* Brb gonna go nap! ***********\n')  




