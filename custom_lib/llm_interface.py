from custom_lib import utils
from custom_lib import console
from custom_lib import dotenvloader

from langchain.schema.document import Document
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
import datetime
import tiktoken
import time
import json

MODEL_4O = "gpt-4o"
MODEL_4O_MINI = "gpt-4o-mini"
MODEL_O1_PREVIEW = "o1-preview"
MODEL_O1_MINI = "o1-mini"

MODEL_TPM = {
    MODEL_4O: 30000,
    MODEL_4O_MINI: 200000,
    MODEL_O1_PREVIEW: 30000,
    MODEL_O1_MINI: 200000
}

MODEL_RPM = {
    MODEL_4O: 500,
    MODEL_4O_MINI: 500,
    MODEL_O1_PREVIEW: 500,
    MODEL_O1_MINI: 500
}

tiktoken_approximate_multiplier = 0.48



def sleep_for_rate_limit(token_count: int, TPM: int, RPM: int):
    # Calculate seconds per token limit
    seconds_per_token = 60 / TPM
    # Calculate the sleep duration for the token limit
    token_limit_duration = token_count * seconds_per_token
    
    # Calculate seconds per request limit
    seconds_per_request = 60 / RPM
    
    # Determine the maximum duration required
    sleep_duration = max(token_limit_duration, seconds_per_request)
    
    # Sleep for the calculated duration
    if sleep_duration > 0:
        console.log(f"Sleeping for {sleep_duration:.2f} seconds to respect rate limits.")
        time.sleep(sleep_duration)

def get_token_count(text, model_name):
    try:
        enc = tiktoken.encoding_for_model(model_name)
    except KeyError:
        # If the model name is unknown to tiktoken, fallback to a known encoding
        console.log(f"tiktoken did not recognize model name: {model_name}")
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def ai_message_to_dict(ai_message):
    """
    Converts an AIMessage object into a dictionary for JSON serialization.
    Handles nested attributes recursively.
    """
    def parse(obj):
        if isinstance(obj, dict):
            return {key: parse(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [parse(item) for item in obj]
        elif hasattr(obj, "__dict__"):  # Handle custom objects
            return {key: parse(value) for key, value in vars(obj).items()}
        else:
            return obj  # Return basic types (int, str, float, None, etc.)

    return parse(ai_message)

def get_timestamp_for_filename():
    now = datetime.datetime.now()
    # Format core date/time as YYYYMMDD-HHMMSS
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    # Extract milliseconds from microseconds
    milliseconds = f"{int(now.microsecond / 1000):03d}"
    # Combine the two parts
    return f"{timestamp}-{milliseconds}"


def get_context_retriever(list_of_strings):
    chunks = []
    for sentence in list_of_strings:
        chunks.append(Document(page_content=sentence))
    console.log(f"context retriever found {len(chunks)} documents")
    if len(chunks) == 0:
        return
    vectorstore = Chroma.from_documents(documents=chunks, embedding = OpenAIEmbeddings())
    llm_context_retriever = vectorstore.as_retriever()
    console.log(f"context retriever updated with {len(chunks)} documents")
    return llm_context_retriever

def debug_final_prompt(template, context_retriever, input_text):
    """
    Returns (not invokes) the final text that would've been sent to the ChatGPT model.
    """
    # 1) Retrieve the documents (if context_retriever is not None)
    if context_retriever is not None:
        docs = context_retriever.get_relevant_documents(input_text)
        # 2) Combine their text into one string
        context_text = "\n".join([doc.page_content for doc in docs])
    else:
        context_text = ""

    # 3) Substitute placeholders in the template
    #    (Your template might have additional placeholders, adapt as needed)
    final_prompt_str = template.replace("{input}", input_text)
    final_prompt_str = final_prompt_str.replace("{context}", context_text)

    return final_prompt_str

llm_output_directory = f'./custom_lib/llm_outputs'
llm_previous_output_directory = f'{llm_output_directory}/previous'
llm_recent_output_path = f'{llm_output_directory}/recent.json'
llm_recent_output_text_path = f'{llm_output_directory}/recent.txt'

def get_natural_text_prompt_and_prediction(result):
    return f'prompt text start: {"_"*50}\n{result["prompt"]}\nprompt text end: {"_"*50}\nprediction text start: {"_"*50}\n{result["prediction"]["content"]}\nprediction text end: {"_"*50}\n\n{json.dumps(result, indent=4)}'

def save_llm_output_to_file(llm_output, current_timestamp):
    utils.create_directory(llm_output_directory)
    utils.create_directory(llm_previous_output_directory)
    utils.delete(llm_recent_output_path)
    utils.delete(llm_recent_output_text_path)
    
    utils.file_dump(llm_output, llm_recent_output_path)
    utils.file_dump(llm_output, f"{llm_previous_output_directory}/{current_timestamp}.json")

    natural_text_output = get_natural_text_prompt_and_prediction(llm_output)
    utils.text_dump(natural_text_output, f"{llm_previous_output_directory}/{current_timestamp}.txt", wrap=True)
    utils.text_dump(natural_text_output, llm_recent_output_text_path, wrap=True)

def llm_cache_directory():
    return f'{llm_output_directory}/cache'

utils.create_directory(llm_cache_directory())

def llm_cache_initialize():
    utils.create_directory(llm_cache_directory())

def llm_cache_reset():
    utils.reset_directory(llm_cache_directory())

def create_cache_key(template, input, model_name, temperature):
    return utils.hash_tuples((template, input, model_name, temperature))

def llm_cache_load(template, input, model_name, temperature):
    key = create_cache_key(template, input, model_name, temperature)
    console.log(f"GPT cache key: {key}")
    cache_file_path = f"{llm_cache_directory()}/{key}.json"
    if not utils.exists(cache_file_path):
        return None
    return utils.file_load(cache_file_path)

def llm_cache_save(template, input, model_name, temperature, return_object):
    key = create_cache_key(template, input, model_name, temperature)
    cache_file_path = f"{llm_cache_directory()}/{key}.json"
    utils.file_dump(return_object, cache_file_path)

def call_gpt_cached(template, run_gpt, input, model_name=MODEL_4O, temperature=0):
    if run_gpt == False:
        return call_gpt(template=template, context_retriever=None, run_gpt=False, input=input, model_name=model_name, temperature=temperature)

    cached_result = llm_cache_load(template=template, input=input, model_name=model_name, temperature=temperature)
    if cached_result is not None:
        console.log("GPT returning cached result")
        return cached_result
    console.log("GPT not returning cached result")
    new_result = call_gpt(template=template, context_retriever=None, run_gpt=run_gpt, input=input, model_name=model_name, temperature=temperature)
    llm_cache_save(template=template, input=input, model_name=model_name, temperature=temperature, return_object=new_result)
    return new_result

def call_gpt(template, context_retriever, run_gpt, input, model_name, temperature):
    current_timestamp = get_timestamp_for_filename()
    return_object = {
        'prompt': None,
        'metadata':{
            'estimated_prompt_token_count': None,
            'percentage_deviation_of_estimated_token_count': None,
            'prediction_time_seconds': None,
            'time_of_call': current_timestamp
        },
        'prediction': None
    }

    final_prompt = debug_final_prompt(template=template, context_retriever=context_retriever, input_text=input)
    return_object['prompt'] = final_prompt
    estimated_prompt_token_count = get_token_count(final_prompt, model_name=model_name)*(1+ tiktoken_approximate_multiplier)
    return_object['metadata']['estimated_prompt_token_count'] = estimated_prompt_token_count
    console.log("prompt to gpt start: "+"_"*50)
    console.log(final_prompt)
    console.log("prompt to gpt end: "+"_"*50)

    if not run_gpt:
        return return_object

    dotenvloader.is_openai_env_set()
    if not "{input}" in template:
        utils.crash_code("template does not contain input slot")
    if not context_retriever is None and not "{context}" in template:
        utils.crash_code("template does not contain context slot")
    prompt = ChatPromptTemplate.from_template(template)

    if model_name == MODEL_O1_PREVIEW or model_name == MODEL_O1_MINI:
        llm = ChatOpenAI(model_name=model_name)
    else:
        llm = ChatOpenAI(model_name = model_name, temperature=temperature)

    if not context_retriever is None:
        rag_chain = (
            {"context": context_retriever,
            "input": RunnablePassthrough()}
            | prompt
            | llm
        )
    else:
        rag_chain = (
            {"input": RunnablePassthrough()}
            | prompt
            | llm
        )

    console.log("gpt call start"+"_"*50)
    start_time = time.time()
    result = rag_chain.invoke(input)
    end_time = time.time()
    console.log("gpt call end"+"_"*50)
    console.log("result from gpt start: "+"_"*50)
    console.log(f"{result}")
    console.log("result from gpt end: "+"_"*50)
    
    return_object['prediction'] = ai_message_to_dict(result)

    exact_prompt_token_count = return_object['prediction']['response_metadata']['token_usage']['prompt_tokens']

    return_object['metadata']['percentage_deviation_of_estimated_token_count'] = abs((exact_prompt_token_count - estimated_prompt_token_count)/estimated_prompt_token_count)
    return_object['metadata']['prediction_time_seconds'] = end_time - start_time

    save_llm_output_to_file(return_object, current_timestamp)
    return return_object

