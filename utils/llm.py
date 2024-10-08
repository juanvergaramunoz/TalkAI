# This file contains the basic methods fo LLM-related tasks.
# References:
# https://platform.openai.com/docs/tutorials/web-qa-embeddings
import openai
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances
import tiktoken

# Load the cl100k_base tokenizer
TOKENIZER = tiktoken.get_encoding("cl100k_base")

def answer_question(
    prompt_guide="Answer the question to the best of your abilities.",
    deployment_name="gpt-4",  # Your deployment name in Azure
    question="Am I allowed to publish model outputs to Twitter, without a human review?",
    context="",
    max_tokens=150,
    stop_sequence=None,
    debug=False
):
    """
    Answer a question using an LLM deployed in Azure OpenAI Service.
    """

    if context:
        context = f" When answering, incorporate the context below.\n\nContext: {context}"

    prompt = f"{prompt_guide}{context}\n\n---\n\nQuestion: {question}\nAnswer:"

    try:
        # Load openai client --> https://github.com/openai/openai-python/blob/main/examples/azure.py
        client = openai.AzureOpenAI(
            api_version=openai.api_version,
            # https://learn.microsoft.com/en-us/azure/cognitive-services/openai/how-to/create-resource?pivots=web-portal#create-a-resource
            azure_endpoint=openai.api_base,
            api_key=openai.api_key,
        )

        # Create a completion using the question and context with the Azure deployment
        response = client.chat.completions.create(
            model=deployment_name,  # e.g. gpt-35-instant
            messages=[
                {
                    "role": "user",
                    #"content": "How do I output all files in a directory using Python?",
                    "content": prompt,
                },
            ],
            #temperature=0,
            max_tokens=max_tokens,
            #top_p=1,
            #frequency_penalty=0,
            #presence_penalty=0,
            #stop=stop_sequence,
        )
        #print(json.loads(response.to_json()))
        response_text = response.choices[0].message.content
            
        
        if debug:
            print(f"Full response: {response}")
        
    except Exception as e:
        print(f"Error: {e}")
        response_text = ""

    print(f"TalkAI answer: {response_text}")
    return response_text


def create_context(question, df_in, max_len=1800, size="ada"):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Create copy of dataframe
    df = df_in.copy()

    # Check if 'memory_embeddings' column exists, if not compute embeddings
    if 'memory_embeddings' not in df.columns:
        # Get the embeddings for the memory
        try:
            df['memory_embeddings'] = df.memory_log.apply(lambda x: openai.embeddings.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])
        except Exception as e:
            print(f"Error generating embeddings for the memory: {e}")
            return ""
        df['n_tokens'] = df.memory_log.apply(lambda x: len(TOKENIZER.encode(x)))
        df['memory_embeddings'] = df['memory_embeddings'].apply(np.array)

    # Get the embeddings for the question
    try:
        q_embeddings = openai.embeddings.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']
    except Exception as e:
        print(f"Error generating embeddings for the question: {e}")
        return ""

    # Compute cosine distances
    df['distances'] = cosine_distances([q_embeddings], np.vstack(df['memory_embeddings'].values)).flatten()

    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for _, row in df.sort_values('distances', ascending=True).iterrows():
        # Calculate the new length
        cur_len += row['n_tokens'] + 4
        
        # If the context is too long, break
        if cur_len > max_len:
            break
        
        # Else add it to the text that is being returned
        returns.append(row["memory_log"])

    # Return the context
    return "\n\n###\n\n".join(returns)


# Function to remove newlines from a pandas series
def remove_newlines(serie):
    serie = serie.str.replace('\n', ' ')
    serie = serie.str.replace('\\n', ' ')
    serie = serie.str.replace('  ', ' ')
    serie = serie.str.replace('  ', ' ')
    return serie