import boto3
from botocore.exceptions import ClientError
import json

# Initialize AWS Bedrock client for general model inference
bedrock = boto3.client(
    service_name=bedrock-runtime,
    region_name=us-west-2  
)

# Initialize Bedrock Knowledge Base client
bedrock_kb = boto3.client(
    service_name=bedrock-agent-runtime,
    region_name=us-west-2
)


def valid_prompt(prompt: str, model_id: str) -> bool:
    """
    Validate if the user prompt is relevant to heavy machinery.
    Returns True if prompt is Category E (heavy machinery), else False.
    """
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""
Human: Classify the provided user request into one of the following categories.
Evaluate the request and return ONLY the category letter (A-E):

Category A: Information about the LLM model or architecture.
Category B: Profanity or toxic content.
Category C: Subject outside heavy machinery.
Category D: Asking about instructions on how you work.
Category E: Only related to heavy machinery.

<user_request>
{prompt}
</user_request>
Assistant:
"""
                    }
                ]
            }
        ]

        response = bedrock.invoke_model(
            modelId=model_id,
            contentType=application/json,
            accept=application/json,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "messages": messages,
                "max_tokens": 10,
                "temperature": 0,
                "top_p": 0.1
            })
        )

        category = json.loads(response[body].read())[content][0]["text"].strip()
        print(f"Prompt category: {category}")

        return category.lower() == "category e"

    except ClientError as e:
        print(f"Error validating prompt: {e}")
        return False


def query_knowledge_base(query: str, kb_id: str):
    """
    Retrieve top 3 results from the Bedrock knowledge base for a given query.
    """
    try:
        response = bedrock_kb.retrieve(
            knowledgeBaseId=kb_id,
            retrievalQuery={text: query},
            retrievalConfiguration={
                vectorSearchConfiguration: {numberOfResults: 3}
            }
        )
        return response.get(retrievalResults, [])
    except ClientError as e:
        print(f"Error querying Knowledge Base: {e}")
        return []


def generate_response(prompt: str, model_id: str, temperature: float = 0.7, top_p: float = 0.9) -> str:
    """
    Generate a response from the Bedrock model given a user prompt.
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
            contentType=application/json,
            accept=application/json,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "messages": messages,
                "max_tokens": 500,
                "temperature": temperature,
                "top_p": top_p
            })
        )

        return json.loads(response[body].read())[content][0]["text"]

    except ClientError as e:
        print(f"Error generating response: {e}")
        return "" 
