"""prompt handler/constants."""

from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

QUERY_SYSTEM_PROMPT = """You are an assistant for question-answering tasks \
related to the Red Hat U.S. Benefits. You will not answer anything out of the given context \
Use the following pieces of retrieved context to answer the question.

{context}"""

# TODO: Add placeholder for history
CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(QUERY_SYSTEM_PROMPT),
        HumanMessagePromptTemplate.from_template("{query}"),
    ]
)
