import gradio as gr
from llama_index.core.agent import ReActAgent
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.mistralai import MistralAI
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from PyPDF2 import PdfReader
def return_hello_with_name(pdf_file) -> str:
    reader = PdfReader(pdf_file)
    page_object = reader.pages[0]
    text = page_object.extract_text()
    with open("/content/drive/MyDrive/data1/data.txt",'w',encoding = 'utf-8') as f:
        f.write(text)

    data_docs = SimpleDirectoryReader("/content/drive/MyDrive/data1").load_data()
    data= VectorStoreIndex.from_documents(data_docs, transformations=[text_splitter])
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
    query_engine_tools = [
    QueryEngineTool(
        query_engine=query_engine1,
        metadata=ToolMetadata(
            name="Create",
            description=(
                "The input is a topic of the operating system. Output is a multiple-choice question and it's answers. Specify the correct answer."
                "Create a multiple-choice question about operating systems."
                "Use other tools to check the answers."

            ),
        ),
    ),
    QueryEngineTool(
        query_engine=query_engine2,
        metadata=ToolMetadata(
            name="Check1",
            description=(
                "The input is a multiple-choice questions. The output is 1 evaluation and 1 multiple-choice question. Specify the correct answer."
                "Conduct a question assessment. Explain the correct answer; if the question or answer is incorrect, make the necessary corrections."
                "If there is no correct answer, then revise the answers."
                "If the answers have the same meaning, then revise them to be correct."
                "Improve the multiple-choice question."
                "The last output is 1 multiple-choice question."


            ),
        ),
    ),
    QueryEngineTool(
        query_engine=query_engine2,
        metadata=ToolMetadata(
            name="Check2",
            description=(
                "The input is a multiple-choice questions. The output is 1 evaluation and 1 multiple-choice question. Specify the correct answer."
                "Conduct a question assessment. Explain the correct answer; if the question or answer is incorrect, make the necessary corrections."
                "If there is no correct answer, then revise the answers."
                "If the answers have the same meaning, then revise them to be correct."
                "Improve the multiple-choice question."
                "The last output is 1 multiple-choice question."

            ),
        ),
    ),

]
    llm = OpenAI(model="gpt-3.5-turbo-16k-0613")
    agent = ReActAgent.from_tools(
      query_engine_tools,
      llm=llm,
      verbose=True,
      context=context,
      )
    return "Đã upload file thành công!"
def response(text, history):
  res = agent.chat(text)
  return str(res)

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Blocks():
              input=gr.File(label="Chọn file")
              output = gr.Textbox(label='Thông báo')
              gr_button = gr.Button('Nhập dữ liệu!')
              gr_button.click(fn=return_hello_with_name, inputs=input , outputs=output, api_name='block_try')
        with gr.Column(scale=4):
            gr.ChatInterface(response)
demo.launch()