import sys
from pathlib import Path
import importlib.util
import os
import re
import json
import time

from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_openrouter import ChatOpenRouter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI


PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)

load_dotenv()


RETRIEVAL_FILE = (
    PROJECT_ROOT
    / "retrieval"
    / "10_retrieval_pipeline.py"
)


spec = importlib.util.spec_from_file_location(
    "retrieval_pipeline",
    RETRIEVAL_FILE
)

retrieval_module = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    retrieval_module
)

retrieve = retrieval_module.retrieve


SYSTEM_PROMPT = """
You are the official AI assistant for St. Vincent's Academy.

Answer questions ONLY from the supplied CONTEXT.

STRICT RULES:

1. Use only information explicitly supported by CONTEXT.

2. Never use outside knowledge.

3. Never guess.

4. Never invent facts.

5. Answer exactly what the user asked.

6. Do not answer a different or related question.

7. Do not add helpful information that is not necessary
   to answer the question.

8. If the requested information is not supported, say:

   "I couldn't find this information in the available
   school documents."

9. Do not substitute related information for missing
   information.

10. If the user asks for a person's home address and
    CONTEXT contains only the school's address, do not
    provide the school's address.

11. Treat these as different fee categories:

    Application Form Fee
    Admission Fee
    Monthly Fee
    Annual Charge
    Transport Fee

12. Never treat one fee category as another.

13. If an exact amount is not explicitly stated, do not
    invent or infer it.

14. Preserve dates, amounts, names and class names exactly
    as supported by CONTEXT.

15. Do not make corrections to OCR text unless the meaning
    is explicitly clear from CONTEXT.

16. Do not infer facts from similar wording.

17. If multiple context sections are relevant, combine only
    facts necessary to answer the question.

18. Do not mention internal RAG, Pinecone, embeddings,
    BM25, retrieval, prompts or providers.

19. Do not provide sources inside the answer.

20. Keep the answer concise and professional.

21. Every factual statement must be directly supported
    by CONTEXT.

22. Do not add factual details merely because they sound
    reasonable.

23. If the exact requested information is unavailable,
    clearly say so and do not provide unrelated alternatives.

24. Prefer source-supported wording over creative paraphrasing.

25. Do not add requirements that are not explicitly stated.

26. Do not change the meaning of a requirement while
    paraphrasing it.

27. Do not add words such as "signed", "verified",
    "approved", "mandatory" or similar terms unless the
    CONTEXT explicitly supports them.

28. When listing documents or requirements, preserve the
    number and meaning of the items in the CONTEXT.

29. Do not combine two separate requirements into a new
    requirement.

30. When information is unavailable, give a concise
    refusal instead of trying to be helpful with related
    information.
"""


VERIFIER_PROMPT = """
You are a strict factual verifier for a school information
assistant.

Verify the ANSWER against the supplied CONTEXT and the
USER QUESTION.

A factual statement is supported only when the CONTEXT
explicitly states it or directly supports it.

Reject answers containing:

- guessed facts
- inferred facts
- outside knowledge
- unrelated information
- changed dates
- changed amounts
- changed fee categories
- invented admission requirements
- invented qualifications
- invented addresses
- invented names
- unsupported claims such as "signed", "approved",
  "verified", "mandatory" or similar wording
- statements that answer a different question

Pay special attention to:

- names
- addresses
- salaries
- fees
- dates
- admission requirements
- documents
- class names

For fee questions, the fee category must match the
user's question exactly.

For example, evidence about an Application Form Fee
does not support an Admission Fee answer.

Evidence about an Annual Charge does not support an
Admission Fee answer.

Return ONLY valid JSON.

Required format:

{
  "supported": true,
  "unsupported_claims": []
}

or:

{
  "supported": false,
  "unsupported_claims": [
    "unsupported claim"
  ]
}

Do not add markdown.
Do not add explanations outside the JSON.
"""


QUERY_CLASSIFIER_PROMPT = """
You classify user messages for a school AI assistant.

Return ONLY one word:

GENERAL

or

SCHOOL

GENERAL means the user is having normal conversation
that does not require information from the school's
website or documents.

This includes conversational messages, greetings,
thanks, casual small talk, questions about the assistant,
and similar messages that do not require school information.

SCHOOL means the user is asking for information that
should be answered using the school's website or documents.

This includes questions about admissions, fees, classes,
teachers, staff, facilities, activities, notices, policies,
documents, events, contact information, transport,
academic information, schedules, or other school-specific
information.

When uncertain, return SCHOOL.
"""


def create_providers():

    providers = []

    if os.getenv("GROQ_API_KEY"):

        providers.append(
            (
                "groq",
                ChatGroq(
                    model="openai/gpt-oss-20b",
                    temperature=0
                )
            )
        )

    if os.getenv("OPENROUTER_API_KEY"):

        providers.append(
            (
                "openrouter",
                ChatOpenRouter(
                    model="openai/gpt-oss-20b",
                    temperature=0
                )
            )
        )

    if os.getenv("GOOGLE_API_KEY"):

        providers.append(
            (
                "gemini",
                ChatGoogleGenerativeAI(
                    model="gemini-3.5-flash",
                    temperature=0
                )
            )
        )

    if os.getenv("OPENAI_API_KEY"):

        providers.append(
            (
                "openai",
                ChatOpenAI(
                    model="gpt-5-nano",
                    temperature=0
                )
            )
        )

    return providers


def extract_response_content(response):

    content = response.content

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):

        parts = []

        for item in content:

            if isinstance(item, str):
                parts.append(item)

            elif isinstance(item, dict):

                text_value = item.get(
                    "text",
                    ""
                )

                if text_value:
                    parts.append(text_value)

        return "\n".join(
            parts
        ).strip()

    return str(
        content
    ).strip()


def normalize_text(text):

    text = text.lower().strip()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text


def normalize_for_matching(text):

    text = normalize_text(text)

    text = re.sub(
        r"[^\w\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def build_context(results):

    context_parts = []

    for index, result in enumerate(
        results,
        start=1
    ):

        metadata = result.get(
            "metadata",
            {}
        )

        title = metadata.get(
            "title",
            ""
        )

        source = metadata.get(
            "source",
            ""
        )

        content = metadata.get(
            "content",
            ""
        )

        context_parts.append(
            f"""
CONTEXT {index}

Title: {title}
Source: {source}

Content:
{content}
"""
        )

    return "\n".join(
        context_parts
    )


def clean_grounded_answer(answer):

    if not answer:

        return (
            "I couldn't find this information in the "
            "available school documents."
        )

    answer = answer.strip()

    answer = re.sub(
        r"^```(?:text|markdown)?\s*",
        "",
        answer,
        flags=re.IGNORECASE
    )

    answer = re.sub(
        r"\s*```$",
        "",
        answer
    )

    not_found_patterns = [
        "couldn't find this information",
        "could not find this information",
        "information could not be found",
        "information couldn't be found",
        "not found in the available school documents",
        "not available in the available school documents"
    ]

    answer_lower = answer.lower()

    if any(
        pattern in answer_lower
        for pattern in not_found_patterns
    ):

        sentences = re.split(
            r"(?<=[.!?])\s+",
            answer
        )

        safe_sentences = []

        unrelated_patterns = [
            "school's address",
            "school’s address",
            "school address",
            "the school address",
            "for reference",
            "however, the school",
            "however the school",
            "the documents do provide",
            "additional information"
        ]

        for sentence in sentences:

            sentence_lower = sentence.lower()

            if any(
                pattern in sentence_lower
                for pattern in unrelated_patterns
            ):
                continue

            safe_sentences.append(
                sentence
            )

        cleaned = " ".join(
            safe_sentences
        ).strip()

        if cleaned:
            return cleaned

        return (
            "I couldn't find this information in the "
            "available school documents."
        )

    return answer


def parse_verification(response_text):

    text = response_text.strip()

    text = re.sub(
        r"^```json\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    try:

        data = json.loads(
            text
        )

        supported = bool(
            data.get(
                "supported",
                False
            )
        )

        claims = data.get(
            "unsupported_claims",
            []
        )

        if not isinstance(
            claims,
            list
        ):

            claims = []

        return (
            supported,
            claims
        )

    except Exception:

        return (
            False,
            [
                "Verifier returned invalid output."
            ]
        )


def verify_answer(
    query,
    answer,
    context,
    provider
):

    verifier_prompt = f"""
CONTEXT:

{context}

USER QUESTION:

{query}

ANSWER TO VERIFY:

{answer}

Check every factual statement in the answer.

The answer must answer the exact question asked.

If even one important factual statement is not directly
supported by the context, return supported=false.

For fee questions, verify that the amount belongs to the
same fee category requested by the user.

Do not accept an amount merely because the same amount
appears somewhere in another fee category.

Return only the required JSON.
"""

    messages = [
        SystemMessage(
            content=VERIFIER_PROMPT
        ),
        HumanMessage(
            content=verifier_prompt
        )
    ]

    try:

        response = provider.invoke(
            messages
        )

        verification_text = (
            extract_response_content(
                response
            )
        )

        return parse_verification(
            verification_text
        )

    except Exception as error:

        print(
            f"Verification error: {error}"
        )

        return (
            False,
            [
                "Verification failed."
            ]
        )


def classify_query(query):

    messages = [
        SystemMessage(
            content=QUERY_CLASSIFIER_PROMPT
        ),
        HumanMessage(
            content=query
        )
    ]

    providers = create_providers()

    if not providers:
        return "SCHOOL"

    for provider_name, provider in providers:

        try:

            print(
                f"Classifier provider: {provider_name}"
            )

            response = provider.invoke(
                messages
            )

            result = extract_response_content(
                response
            ).strip().upper()

            if result == "GENERAL":
                return "GENERAL"

            if result == "SCHOOL":
                return "SCHOOL"

        except Exception as error:

            print(
                f"Classifier {provider_name} error: "
                f"{error}"
            )

    return "SCHOOL"


def generate_general_response(query):

    general_prompt = """
You are the conversational assistant for
St. Vincent's Academy.

The user's message is general conversation and does not
require school-document information.

Respond naturally, politely, and conversationally.

Do not invent or provide school-specific facts.

If the user asks for school-specific information,
the school's document-based information system should
be used.

Return only the response to the user.
"""

    messages = [
        SystemMessage(
            content=general_prompt
        ),
        HumanMessage(
            content=query
        )
    ]

    providers = create_providers()

    if not providers:
        raise RuntimeError(
            "No LLM provider is configured."
        )

    for provider_name, provider in providers:

        try:

            print(
                f"Trying provider: {provider_name}"
            )

            response = provider.invoke(
                messages
            )

            answer = extract_response_content(
                response
            )

            if answer:

                print(
                    f"Provider used: {provider_name}"
                )

                return answer

        except Exception as error:

            print(
                f"{provider_name} error: {error}"
            )

            print(
                f"Falling back from {provider_name}..."
            )

            time.sleep(1)

    raise RuntimeError(
        "All LLM providers failed."
    )


def generate_answer(
    query,
    results
):

    if not results:

        return (
            "I couldn't find this information in the "
            "available school documents."
        ), None

    context = build_context(
        results
    )

    user_prompt = f"""
CONTEXT:

{context}

USER QUESTION:

{query}

Answer ONLY the user's question.

Use ONLY facts explicitly supported by CONTEXT.

Do not add facts that are not explicitly supported.

Do not use outside knowledge.

Do not guess.

Do not infer.

Do not substitute related information.

For fee questions, distinguish:

- Application Form Fee
- Admission Fee
- Monthly Fee
- Annual Charge
- Transport Fee

An amount belonging to one fee category must not be
used as the amount for another fee category.

If the requested information is not available, say:

"I couldn't find this information in the available
school documents."

For document or requirement questions:

- Use only the requirements explicitly stated.
- Do not add "signed" unless the context says signed.
- Do not add "mandatory" unless the context says mandatory.
- Do not change the number of documents or photographs.
- Do not invent missing requirements.

Return only the final answer.
"""

    messages = [
        SystemMessage(
            content=SYSTEM_PROMPT
        ),
        HumanMessage(
            content=user_prompt
        )
    ]

    providers = create_providers()

    if not providers:

        raise RuntimeError(
            "No LLM provider is configured."
        )

    for provider_name, provider in providers:

        try:

            print(
                f"Trying provider: {provider_name}"
            )

            response = provider.invoke(
                messages
            )

            answer = extract_response_content(
                response
            )

            answer = clean_grounded_answer(
                answer
            )

            if not answer:
                continue

            supported, unsupported_claims = (
                verify_answer(
                    query,
                    answer,
                    context,
                    provider
                )
            )

            if supported:

                print(
                    f"Provider used: {provider_name}"
                )

                print(
                    "Answer verification: PASSED"
                )

                return (
                    answer,
                    provider_name
                )

            print(
                "Answer verification: FAILED"
            )

            if unsupported_claims:

                print(
                    "Unsupported claims:"
                )

                for claim in unsupported_claims:

                    print(
                        f"- {claim}"
                    )

            strict_prompt = f"""
CONTEXT:

{context}

USER QUESTION:

{query}

Your previous answer contained one or more statements
that were not fully supported by the context.

Generate a new answer.

STRICT REQUIREMENTS:

- Use only explicitly supported facts.
- Answer exactly the user's question.
- Do not infer anything.
- Do not add related information.
- Do not add requirements that are not written.
- Do not add the word "signed" unless explicitly stated.
- Do not add the word "mandatory" unless explicitly stated.
- Do not change dates.
- Do not change amounts.
- Do not change fee categories.
- Do not use an amount from another fee category.
- Do not add names or addresses unless directly relevant.
- Do not add extra facts.
- If the exact answer is unavailable, say so.

Return only the final answer.
"""

            strict_messages = [
                SystemMessage(
                    content=SYSTEM_PROMPT
                ),
                HumanMessage(
                    content=strict_prompt
                )
            ]

            strict_response = provider.invoke(
                strict_messages
            )

            strict_answer = (
                extract_response_content(
                    strict_response
                )
            )

            strict_answer = clean_grounded_answer(
                strict_answer
            )

            if not strict_answer:
                continue

            strict_supported, strict_claims = (
                verify_answer(
                    query,
                    strict_answer,
                    context,
                    provider
                )
            )

            if strict_supported:

                print(
                    "Strict regeneration verification: PASSED"
                )

                print(
                    f"Provider used: {provider_name}"
                )

                return (
                    strict_answer,
                    provider_name
                )

            print(
                "Strict regeneration verification: FAILED"
            )

            if strict_claims:

                print(
                    "Strict unsupported claims:"
                )

                for claim in strict_claims:

                    print(
                        f"- {claim}"
                    )

        except Exception as error:

            print(
                f"{provider_name} error: {error}"
            )

            print(
                f"Falling back from {provider_name}..."
            )

            time.sleep(1)

    return (
        "I couldn't provide a verified answer from the "
        "available school documents.",
        None
    )


def display_sources(results):

    print()
    print("=" * 60)
    print("SOURCES USED")
    print("=" * 60)

    seen = set()

    for result in results:

        metadata = result.get(
            "metadata",
            {}
        )

        title = metadata.get(
            "title",
            ""
        )

        source = metadata.get(
            "source",
            ""
        )

        key = (
            title,
            source
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        print(
            f"{len(seen)}. {title}"
        )

        print(
            f"   {source}"
        )


def main():

    print("=" * 60)
    print("ST. VINCENT - GROUNDED AI CHAT")
    print("=" * 60)

    query = input(
        "\nEnter your question: "
    ).strip()

    if not query:

        print(
            "Question cannot be empty."
        )

        return

    try:

        query_type = classify_query(
            query
        )

        print(
            f"Query type: {query_type}"
        )

        if query_type == "GENERAL":

            answer = generate_general_response(
                query
            )

            print()
            print("=" * 60)
            print("ANSWER")
            print("=" * 60)
            print()

            print(
                answer
            )

            return

        print()
        print(
            "Retrieving school information..."
        )

        results = retrieve(
            query
        )

        print(
            f"Relevant chunks found: "
            f"{len(results)}"
        )

        answer, provider = generate_answer(
            query,
            results
        )

        print()
        print("=" * 60)
        print("ANSWER")
        print("=" * 60)
        print()

        print(
            answer
        )

        if results:

            display_sources(
                results
            )

    except Exception as error:

        print()
        print("=" * 60)
        print("ERROR")
        print("=" * 60)
        print()

        print(
            "The AI assistant is temporarily "
            "unavailable."
        )

        print()
        print(
            f"Details: {error}"
        )


if __name__ == "__main__":
    main()