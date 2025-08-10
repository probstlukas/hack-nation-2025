import openai
from util.create_vector_db import create_chroma_vector_db


def answer_question(question: str, pdf_path: str):
    vector_db = create_chroma_vector_db(pdf_path)

    query = question
    response = openai.embeddings.create(input=[query], model="text-embedding-3-small")

    query_result = vector_db.query(
        query_embeddings=[response.data[0].embedding], n_results=10
    )["documents"][0]
    context = "Document Fragement:\n".join(query_result)

    chat_response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": f"Plase anwer the question '{query}' based one of the following document fragments (the answer is probably in there): Context: {context}",
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
