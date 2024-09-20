# Description: Generic functions used in the project. Includes transformations, embeddings, and other functions.
# References:
# https://platform.openai.com/docs/tutorials/web-qa-embeddings
import tiktoken

# Load the cl100k_base tokenizer which is designed to work with the ada-002 model
TOKENIZER = tiktoken.get_encoding("cl100k_base")

# Function to split the text into chunks of a maximum number of tokens
def split_text_based_on_token_length(text, max_tokens=500):
    print(f"Splitting text into chunks of {max_tokens} tokens")
    print(f"Text: {text}")
    
    # Split the text into sentences
    sentences = text.split('. ')
    
    # Get the number of tokens for each sentence
    n_tokens = [len(TOKENIZER.encode(" " + sentence)) for sentence in sentences]
    
    chunks = []
    tokens_so_far = 0
    chunk = []

    # Loop through the sentences and tokens joined together in a tuple
    for sentence, token in zip(sentences, n_tokens):
        # If the current sentence alone is longer than max tokens, split it further
        if token > max_tokens:
            long_sentence_chunks = split_long_sentence(sentence, max_tokens)
            for sub_chunk in long_sentence_chunks:
                if tokens_so_far + len(TOKENIZER.encode(" " + sub_chunk)) > max_tokens:
                    chunks.append(". ".join(chunk) + ".")
                    chunk = []
                    tokens_so_far = 0
                chunk.append(sub_chunk)
                tokens_so_far += len(TOKENIZER.encode(" " + sub_chunk)) + 1
            continue

        # If the number of tokens so far plus the number of tokens in the current sentence 
        # is greater than the max number of tokens, then add the chunk to the list of chunks 
        # and reset the chunk and tokens so far
        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0

        # Otherwise, add the sentence to the chunk and add the number of tokens to the total
        chunk.append(sentence)
        tokens_so_far += token + 1

    # Add the last chunk to the list of chunks
    if chunk:
        chunks.append(". ".join(chunk) + ".")

    return chunks

# Helper function to split long sentences
def split_long_sentence(sentence, max_tokens):
    words = sentence.split()
    sub_chunk = []
    sub_token_count = 0
    sub_chunks = []

    for word in words:
        word_token_count = len(TOKENIZER.encode(" " + word))
        # Check if adding this word exceeds the max token limit
        if sub_token_count + word_token_count > max_tokens:
            # Create a sub-chunk and reset counts
            sub_chunks.append(" ".join(sub_chunk))
            sub_chunk = [word]
            sub_token_count = word_token_count
        else:
            sub_chunk.append(word)
            sub_token_count += word_token_count

    # Add the last sub-chunk if it has content
    if sub_chunk:
        sub_chunks.append(" ".join(sub_chunk))
    
    return sub_chunks