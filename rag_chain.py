# import os
# from langchain_groq import ChatGroq
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_core.prompts import PromptTemplate
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains import LLMChain
# from langchain.schema.runnable import RunnableMap

# from dotenv import load_dotenv
# load_dotenv()

# DB_PATH = "vector_store"

# def load_rag_chain():
#     # Embeddings
#     embeddings = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2"
#     )

#     # Load FAISS
#     vectorstore = FAISS.load_local(
#         DB_PATH,
#         embeddings,
#         allow_dangerous_deserialization=True
#     )

#     retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

#     # LLM
#     llm = ChatGroq(
#     groq_api_key=os.getenv("GROQ_API_KEY"),
#     model_name="llama-3.1-8b-instant",
#     temperature=0.2
# )


#     # Prompt
#     prompt = PromptTemplate(
#         input_variables=["context", "question"],
#         template=(
#             "Use ONLY the provided context to answer the question.\n\n"
#             "Context:\n{context}\n\n"
#             "Question:\n{question}\n\n"
#             "Answer:"
#         ),
#     )

#     # Chain to generate final answer
#     document_chain = create_stuff_documents_chain(
#         llm=llm,
#         prompt=prompt
#     )

#     # FULL RAG PIPELINE (manual, stable, no 'input' bug)
#     rag_chain = (
#     RunnableMap({
#         "context": lambda x: retriever.invoke(x["question"]),
#         "question": lambda x: x["question"],
#     })
#     | document_chain
# )


#     return rag_chain


import os
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings  # ← CHANGED
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema.runnable import RunnableMap

from dotenv import load_dotenv
load_dotenv()

DB_PATH = "vector_store"

def load_rag_chain():
    # Embeddings — lightweight, ~50MB RAM instead of 500MB
    embeddings = FastEmbedEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"   # ← CHANGED
    )

    # Load FAISS — everything below is UNCHANGED
    vectorstore = FAISS.load_local(
        DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0.2
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "Use ONLY the provided context to answer the question.\n\n"
            "Context:\n{context}\n\n"
            "Question:\n{question}\n\n"
            "Answer:"
        ),
    )

    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
    )

    rag_chain = (
        RunnableMap({
            "context": lambda x: retriever.invoke(x["question"]),
            "question": lambda x: x["question"],
        })
        | document_chain
    )

    return rag_chain