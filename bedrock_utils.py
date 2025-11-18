import boto3
from botocore.exceptions import ClientError
import json

# Bedrock Runtime (invoke Claude / Titan models)
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-west-2"
)

# Bedrock Knowledge Base Runtime
bedrock_kb = boto3.client(
    service_name="bedrock-agent-runtime",
    region_name="us-west-2"
)


# ---------------------------------------------------------
# 1. VALIDATE PROMPT (Category E = Heavy Machinery)
# ---------------------------------------------------------
def valid_prompt(prompt: str, model_id: str) -> bool:
    """
    Validate whether the user prompt belongs to Category E (heavy machinery).
    Returns True only if the LLM returns exactly Category E.
    """
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""
Human: Classify the user request into EXACTLY one of the categories below.
Return ONLY the category letter (A–E).

A = Asking about how the LLM works or architecture 
B = Profanity / toxic 
C = Not about heavy machinery 
D = Asking how you work or your system instructions 
E = ONLY about heavy machinery

<user_request>{prompt}</user_request>

Assistant:"""
                    }
                ]
            }
        ]

        response = bedrock.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "messages": messages,
                "max_tokens": 10,
                "temperature": 0,
                "top_p": 0.1
            })
        )

        result = json.loads(response["body"].read())
        category = result["content"][0]["text"].strip()

        print("LLM classified prompt as:", category)

        return category.lower() == "category e"

    except ClientError as e:
        print(f"Error validating prompt: {e}")
        return False


# ---------------------------------------------------------
# 2. QUERY KNOWLEDGE BASE (Correct Bedrock KB Retrieve API)
# ---------------------------------------------------------
def query_knowledge_base(query: str, kb_id: str):
    """
    Query the Bedrock Knowledge Base using semantic vector search.
    Returns the list of top 3 retrieval results.
    """
    try:
        response = bedrock_kb.retrieve(
            knowledgeBaseId=kb_id,
            retrievalQuery={"text": query},
            retrievalConfiguration={
                "vectorSearchConfiguration": {
                    "numberOfResults": 3
                }
            }
        )

        return response.get("retrievalResults", [])

    except ClientError as e:
        print(f"Error querying Knowledge Base: {e}")
        return []


# ---------------------------------------------------------
# 3. GENERATE RESPONSE FROM BEDROCK MODEL
# ---------------------------------------------------------
def generate_response(prompt: str, model_id: str, temperature: float = 0.7, top_p: float = 0.9) -> str:
    """
    Generate text output from the Bedrock model using Claude format.
    """
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        response = bedrock.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "messages": messages,
                "max_tokens": 500,
                "temperature": temperature,
                "top_p": top_p
            })
        )

        result = json.loads(response["body"].read())
        return result["content"][0]["text"]

    except ClientError as e:
        print(f"Error generating response: {e}")
        return "" 
