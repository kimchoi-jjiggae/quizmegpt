# quizmegpt

This project is a fun way to generate short-answer test questions based on a given document and evaluate a student's response to the question by giving feedback. It uses various language models and algorithms to perform the following tasks:

- Load a document from a PDF file using PyPDFLoader
-Split the document into chunks using CharacterTextSplitter
-Extract embeddings from the text using OpenAIEmbeddings
-Create a vector store using Chroma
-Implement a RetrievalQA chain using OpenAI and the vector store as the index
-Generate a question based on a specific document chunk using OpenAI's GPT model
-Evaluate the student's answer to the generated question and give feedback using OpenAI's GPT model
-The project uses the OpenAI API to interact with GPT models and requires an API key to function. The API key is loaded from a .env file in the project directory using load_dotenv() and stored in the api_key variable.

To use the project, simply provide a PDF file path and run the script. It will generate a short-answer test question based on a specific chunk of the document and prompt the user to provide an answer. It will then evaluate the answer and provide feedback to the user.

Note that this project is purely for fun and should not be used for any serious educational purposes.

Enjoy the project!

Limitations: Right now it only will parse ~1-2 pages of your document, since OpenAI has restrictions on how much data you can upload to get a response
