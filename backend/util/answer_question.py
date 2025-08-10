from typing import List
import openai
from typing import Optional
from .create_vector_db import create_chroma_vector_db


def general_promot_no_context(question: str):
    chat_response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": question,
            },
        ],
        max_tokens=300,
        temperature=0.2,
    )
    answer = chat_response.choices[0].message.content
    return answer


def answer_question_openai(
    question: str, pdf_path: str, top_k: int, extra_contex: Optional[str] = None
):
    vector_db = create_chroma_vector_db(pdf_path)

    query = question
    response = openai.embeddings.create(input=[query], model="text-embedding-3-small")

    query_result = vector_db.query(
        query_embeddings=[response.data[0].embedding], n_results=top_k
    )["documents"][0]
    context = "Document Fragement:\n".join(query_result)

    if extra_contex is not None:
        context = extra_contex + "\n" + context

    promt = f"Please serve the promt '{query}' based one of the following document fragments (the answer is probably in there): Context: {context}"

    print(context)

    chat_response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": promt,
            },
        ],
        max_tokens=300,
        temperature=0.2,
    )

    answer = chat_response.choices[0].message.content
    return answer


if __name__ == "__main__":
    a = answer_question(
        "What is on page 5?", "datasets/financebench/pdfs/3M_2015_10K.pdf"
    )
    print(a)
