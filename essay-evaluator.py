# This workflow evaluates an essay across three parallel aspects (`language`, `analysis`, `thought`) using Pydantic structured output, aggregates the scores into a list using `operator.add` as a reducer, and generates an overall summary and average score.
import operator
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

load_dotenv()

# Initialize Model
model = ChatOpenAI(model="gpt-4o-mini")

# Define Pydantic Schema for Structured Output
class EvaluationSchema(BaseModel):
    feedback: str = Field(description="Detailed feedback for the essay")
    score: int = Field(description="Score out of 10", ge=0, le=10)

structured_model = model.with_structured_output(EvaluationSchema)

# Define State with Reducer Function
class UPSCState(TypedDict):
    essay: str
    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str
    overall_feedback: str
    individual_scores: Annotated[List[int], operator.add]
    average_score: float

# Define Node Functions
def evaluate_language(state: UPSCState):
    prompt = f"Evaluate the language quality of the following essay and provide a feedback and assign a score out of 10:\n\n{state['essay']}"
    output = structured_model.invoke(prompt)
    return {
        "language_feedback": output.feedback,
        "individual_scores": [output.score]
    }

def evaluate_analysis(state: UPSCState):
    prompt = f"Evaluate the depth of analysis of the following essay and provide a feedback and assign a score out of 10:\n\n{state['essay']}"
    output = structured_model.invoke(prompt)
    return {
        "analysis_feedback": output.feedback,
        "individual_scores": [output.score]
    }

def evaluate_thought(state: UPSCState):
    prompt = f"Evaluate the clarity of thought of the following essay and provide a feedback and assign a score out of 10:\n\n{state['essay']}"
    output = structured_model.invoke(prompt)
    return {
        "clarity_feedback": output.feedback,
        "individual_scores": [output.score]
    }

def final_evaluation(state: UPSCState):
    prompt = f"""Based on the following feedbacks create a summarized feedback:
Language Feedback: {state['language_feedback']}
Depth of Analysis Feedback: {state['analysis_feedback']}
Clarity of Thought Feedback: {state['clarity_feedback']}"""

    response = model.invoke(prompt)
    overall_feedback = response.content

    avg_score = sum(state["individual_scores"]) / len(state["individual_scores"])

    return {
        "overall_feedback": overall_feedback,
        "average_score": avg_score
    }

# Build Graph
graph = StateGraph(UPSCState)

# Add Nodes
graph.add_node("evaluate_language", evaluate_language)
graph.add_node("evaluate_analysis", evaluate_analysis)
graph.add_node("evaluate_thought", evaluate_thought)
graph.add_node("final_evaluation", final_evaluation)

# Add Edges from START to Parallel Evaluators
graph.add_edge(START, "evaluate_language")
graph.add_edge(START, "evaluate_analysis")
graph.add_edge(START, "evaluate_thought")

# Connect Parallel Evaluators to Final Evaluation
graph.add_edge("evaluate_language", "final_evaluation")
graph.add_edge("evaluate_analysis", "final_evaluation")
graph.add_edge("evaluate_thought", "final_evaluation")

# Connect Final Evaluation to END
graph.add_edge("final_evaluation", END)

# Compile Workflow
workflow = graph.compile()

# Execute Workflow
sample_essay = "India's role in AI development and global tech..."

initial_state = {
    "essay": sample_essay
}

result = workflow.invoke(initial_state)
print(result)
