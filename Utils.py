from openai import OpenAI

def query_chatgpt(prompt, model_type="gpt-4", max_tokens=1000, api_key="YOUR_API_KEY", retries=3):
    """
    Queries ChatGPT-like API with a prompt and returns the response, retrying in case of throttling errors.

    Parameters:
        prompt (str): The input prompt to send to the model.
        model_type (str): The type of model to use (default is "gpt-4").
        max_tokens (int): Maximum number of tokens in the response (default is 1000).
        api_key (str): Your OpenAI API key.
        retries (int): Number of times to retry in case of throttling errors (default is 3).

    Returns:
        str: The model's response to the input prompt or an error message after retries.
    """

        # Set your OpenAI API key
    client = OpenAI(
        api_key=api_key,
    )

    response = client.with_options(max_retries=retries).chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        max_tokens=max_tokens,
        model=model_type,
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    # Example usage
    prompt = "Explain the concept of reinforcement learning."
    response = query_chatgpt(prompt, model_type="gpt-4o", max_tokens=500, api_key="YOUR_API_KEY", retries=5)
    print(response)