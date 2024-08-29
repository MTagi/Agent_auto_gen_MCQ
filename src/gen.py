
from llama_index.llms.openai import OpenAI

import os
os.environ['OPENAI_API_KEY'] =""
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)


from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=10)

# global
from llama_index.core import Settings

Settings.text_splitter = text_splitter

data_docs = SimpleDirectoryReader(
        "/content/drive/MyDrive/data"
    ).load_data()
data= VectorStoreIndex.from_documents(data_docs, transformations=[text_splitter])

from llama_index.core import PromptTemplate
PROMPT_TEMPLATE1 = (
    "Dưới đây là một chủ đề về môn hệ điều hành."
    "\n -----------------------------\n"
    "{context_str}"
    "\n -----------------------------\n"
    "Bạn là một chuyên gia về hệ điều hành và là giảng viên để tạo ra các câu hỏi trắc nghiệm về bộ môn hệ điều hành, hãy sinh ra câu hỏi trắc nghiệm mới dựa trên chủ đề đưa vào, có nêu đáp án đúng."
    "Tạo câu hỏi về chủ đề là:  {query_str}"
)
QA_PROMPT1 = PromptTemplate(PROMPT_TEMPLATE1)
# Query Engine
query_engine1 = data.as_query_engine( similarity_top_k=3,
                                     #similarity_top_k=3 / Find (3) most similarity answer like question
                                    text_qa_template=QA_PROMPT1,
                                    # Prompt Input
                                    llm = OpenAI(model='gpt-3.5-turbo-16k-0613' ,temperature=0.5, max_tokens=512),
                                    #Using OpenAI LLM to service context for output answer
                                    max_tokens=-1
                                    #Max length of answer (-1 is nonelimits length)
                                    )
from llama_index.core import PromptTemplate
PROMPT_TEMPLATE2 = (
    "Đầu vào là 1 câu hỏi trắc nghiệm về bộ môn hệ điều hành."
    "\n -----------------------------\n"
    "{context_str}"
    "\n -----------------------------\n"
    "Bạn là một chuyên gia về hệ điều hành, hãy kiểm tra lại độ chính xác của câu hỏi và chỉnh sửa lại chúng tốt hơn."
    " Đầu ra là 1 lời đánh giá và một câu hỏi trắc nghiệm. Và chỉ rõ đáp án đúng của nó. Hãy đánh giá và cập nhật câu hỏi:  {query_str}"
    "Đảm bảo định dạng phản hồi là 1 lời đánh giá và 1câu hỏi trắc nghiệm có 4 đáp án lựa chọn có chỉ rõ đáp án đúng"
)
QA_PROMPT2 = PromptTemplate(PROMPT_TEMPLATE2)

# Query Engine
query_engine2 = data.as_query_engine( similarity_top_k=3,
                                     #similarity_top_k=3 / Find (3) most similarity answer like question
                                    text_qa_template=QA_PROMPT2,
                                    # Prompt Input
                                    llm = OpenAI(model='gpt-3.5-turbo-16k-0613' ,temperature=0.1, max_tokens=512),
                                    #Using OpenAI LLM to service context for output answer
                                    max_tokens=-1
                                    #Max length of answer (-1 is nonelimits length)
                                    )

