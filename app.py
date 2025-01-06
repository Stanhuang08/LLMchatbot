import os
import uvicorn
from fastapi import FastAPI
from model.input import Input
from data_storage.data_store import DataStore
from model.llama import GetAnswerGPT3
from llama_index import (
    LLMPredictor,
    PromptHelper,
    ServiceContext,
    StorageContext, 
    load_index_from_storage
)
# from auth import get_user
from fastapi import APIRouter, Depends


from llama_index.llms import AzureOpenAI
from llama_index.embeddings import AzureOpenAIEmbedding

PORT = 5678
indexDir = "index-ei/"

api_version3 = "2023-03-15-preview"
azure_endpoint3 = "https://ltc-to-openai.openai.azure.com/"
api_key3 = "0f1a03e98a2046218282e875964f5cc6"

max_input_size = 4096
num_output = 1024
max_chunk_overlap = 1
topkrelevant = 1

llm = AzureOpenAI(
    temperature = 0, 
    max_tokens = 4096,
    deployment_name="gpt-35-turbo-0301", 
    model = "gpt-35-turbo",
#    deployment_name = "text-davinci-003", 
#    model = "text-davinci-003",
    api_key = api_key3,
    azure_endpoint = azure_endpoint3,
    api_version = api_version3,    
)

llm_predictor = LLMPredictor(llm=llm)
embed_model = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name="text-embedding-ada-002",
    api_key=api_key3,
    azure_endpoint=azure_endpoint3,
    api_version=api_version3,
)
prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)
service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor,
    embed_model= embed_model,
    prompt_helper=prompt_helper
)
storage_context = StorageContext.from_defaults(persist_dir=indexDir)
INDEX = load_index_from_storage(storage_context, service_context=service_context)

# app = FastAPI(
#     title="Customer Service Q&A Chatbot",
#     version="0.1.0",
#     root_path="/chatbot_service",
#     docs_url="/swagger",
#     redoc_url="/swagger-redoc",
#     openapi_url="/chatbot_service/openapi.json"
#     )

app = FastAPI(docs_url="/chatbot_service/docs",
              openapi_url="/chatbot_service/openapi.json")
router = APIRouter(prefix="/chatbot_service")

# router = APIRouter(
#     prefix="/chatbot_service",
#     tags=["chatbot_service"],
#     responses={404: {"description": "Not found"}},
# )

# @app.get('/')
@router.get('/')
#def index(user: dict = Depends(get_user)):
def index():
    return {'message': 'Customer service Q&A chatbot is running'}

# @app.post('/answergenerate')
@router.post('/answergenerate')
#def answer_generate(input: Input, user: dict = Depends(get_user)):
def answer_generate(input: Input):
    answer = GetAnswerGPT3(INDEX, input.input_text, topkrelevant)
    # DataStore().store_data(input.site_id, input.user_id, input.input_text, answer)
    return {'output': answer}
    

app.include_router(router)

if __name__ == '__main__':
    print('docker image successful running')
    # uvicorn.run(app, host='127.0.0.1', port=PORT)
    # uvicorn.run(app, host='10.2.26.50/', port=PORT)
    uvicorn.run(app, host='', port=PORT)