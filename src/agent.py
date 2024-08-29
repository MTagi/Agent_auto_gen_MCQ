from llama_index.core.agent import ReActAgent
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.mistralai import MistralAI
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.core.tools import QueryEngineTool, ToolMetadata
query_engine_tools = [
    QueryEngineTool(
        query_engine=query_engine1,
        metadata=ToolMetadata(
            name="Create",
            description=(
                "The input is a requirement. Output is a multiple-choice question and it's answers. Specify the correct answer. Create a multiple-choice question about operating systems."
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
