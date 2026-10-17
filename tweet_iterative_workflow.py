# Here is the complete Python code discussed in the video **"Iterative Workflows in LangGraph | Agentic AI using LangGraph | Video 8 | CampusX"**. 

# The workflow creates an **iterative/looping agentic system** that generates tweets, evaluates them using a critic LLM, and refines them via an optimizer LLM until the tweet meets approval criteria or reaches the maximum iteration limit.


from typing import TypedDict, Literal, List
import operator
from typing_extensions import Annotated
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# 1. Initialize LLMs
# ---------------------------------------------------------------------------
# In an ideal production setup, different models optimized for generation, 
# evaluation, and optimization are used. In the video, GPT-4o-mini is used.
generator_llm = ChatOpenAI(model="gpt-4o-mini")
evaluator_llm = ChatOpenAI(model="gpt-4o-mini")
optimizer_llm = ChatOpenAI(model="gpt-4o-mini")

# ---------------------------------------------------------------------------
# 2. Define State Schema
# ---------------------------------------------------------------------------
# The state keeps track of the topic, current tweet, evaluation result, feedback,
# iteration count, max allowed iterations, and accumulated histories.
class TweetState(TypedDict):
    topic: str
    tweet: str
    evaluation: Literal["approved", "needs_improvement"]
    feedback: str
    iteration: int
    max_iteration: int
    tweet_history: Annotated[List[str], operator.add]
    feedback_history: Annotated[List[str], operator.add]

# ---------------------------------------------------------------------------
# 3. Define Pydantic Schema for Evaluator Structured Output
# ---------------------------------------------------------------------------
# Forces the evaluator LLM to return a structured JSON response.
class TweetEvaluationSchema(BaseModel):
    evaluation: Literal["approved", "needs_improvement"] = Field(
        description="Evaluation result: either 'approved' or 'needs_improvement'"
    )
    feedback: str = Field(
        description="Detailed feedback describing strengths and weaknesses of the tweet"
    )

# ---------------------------------------------------------------------------
# 4. Define Node Functions
# ---------------------------------------------------------------------------

def generate_tweet(state: TweetState):
    """Generates the initial tweet based on the topic."""
    messages = [
        SystemMessage(content="You are a funny and clever Twitter influencer."),
        HumanMessage(
            content=f"Write a short, original, and hilarious tweet on the topic: {state['topic']}.\n"
                    "Rules: Do not use Q&A format. Stay under 280 characters. "
                    "Use observational humor, irony, sarcasm, or cultural references. "
                    "Think in meme logic, punchlines, and relatable takes. Use simple day-to-day English."
        )
    ]
    response = generator_llm.invoke(messages).content
    return {
        "tweet": response,
        "tweet_history": [response]
    }

def evaluate_tweet(state: TweetState):
    """Evaluates the tweet against strict criteria using structured output."""
    structured_evaluator = evaluator_llm.with_structured_output(TweetEvaluationSchema)
    messages = [
        SystemMessage(
            content="You are a ruthless, no-laughs-given Twitter critic. "
                    "You evaluate tweets based on humor, originality, virality, and tweet format."
        ),
        HumanMessage(
            content=f"Evaluate the following tweet: {state['tweet']}\n"
                    f"Topic: {state['topic']}\n"
                    "Criteria: Check for originality, humor, punchiness, virality potential, and format. "
                    "Automatically reject if it is Q&A format, over 280 characters, or traditional jokes."
        )
    ]
    response = structured_evaluator.invoke(messages)
    return {
        "evaluation": response.evaluation,
        "feedback": response.feedback,
        "feedback_history": [response.feedback]
    }

def optimize_tweet(state: TweetState):
    """Rewrites and optimizes the tweet based on evaluator feedback."""
    messages = [
        SystemMessage(content="You punch up tweets for virality and humor based on given feedback."),
        HumanMessage(
            content=f"Improve the tweet based on this feedback: {state['feedback']}\n"
                    f"Topic of the tweet: {state['topic']}\n"
                    f"Original tweet: {state['tweet']}\n"
                    "Rewrite it as a short, viral-worthy tweet. Avoid Q&A style and stay under 280 characters."
        )
    ]
    response = optimizer_llm.invoke(messages).content
    current_iteration = state.get("iteration", 1) + 1
    return {
        "tweet": response,
        "iteration": current_iteration,
        "tweet_history": [response]
    }

# ---------------------------------------------------------------------------
# 5. Define Routing Decision Function
# ---------------------------------------------------------------------------

def route_evaluation(state: TweetState):
    """Determines whether to end the graph or loop back for optimization."""
    if state["evaluation"] == "approved" or state["iteration"] >= state["max_iteration"]:
        return "approved"
    else:
        return "needs_improvement"

# ---------------------------------------------------------------------------
# 6. Build the LangGraph Workflow
# ---------------------------------------------------------------------------

graph = StateGraph(TweetState)

# Add Nodes
graph.add_node("generate", generate_tweet)
graph.add_node("evaluate", evaluate_tweet)
graph.add_node("optimize", optimize_tweet)

# Add Linear Edges
graph.add_edge(START, "generate")
graph.add_edge("generate", "evaluate")

# Add Conditional Edge from Evaluator
graph.add_conditional_edges(
    "evaluate",
    route_evaluation,
    {
        "approved": END,
        "needs_improvement": "optimize"
    }
)

# Loop back from Optimizer to Evaluator
graph.add_edge("optimize", "evaluate")

# Compile Graph
workflow = graph.compile()

# ---------------------------------------------------------------------------
# 7. Execute the Workflow
# ---------------------------------------------------------------------------

initial_state = {
    "topic": "Indian Railways",
    "iteration": 1,
    "max_iteration": 5
}

result = workflow.invoke(initial_state)

# Display final results and execution histories
print("=== FINAL APPROVED TWEET ===")
print(result.get("tweet"))

print("\n=== TWEET REVISION HISTORY ===")
for i, tweet in enumerate(result.get("tweet_history", []), 1):
    print(f"Iteration {i}: {tweet}")

print("\n=== FEEDBACK HISTORY ===")
for i, fb in enumerate(result.get("feedback_history", []), 1):
    print(f"Feedback {i}: {fb}")

### Key Workflow Components Summary
# 1. **Generator Node (`generate`)**: Takes the user topic and crafts an initial tweet.
# 2. **Evaluator Node (`evaluate`)**: Uses `with_structured_output` and a Pydantic schema (`TweetEvaluationSchema`) to rate the tweet strictly as `approved` or `needs_improvement` along with constructive feedback.
# 3. **Optimizer Node (`optimize`)**: Receives the previous draft and feedback, refines the tweet, and increments the `iteration` counter.
# 4. **Conditional Routing (`route_evaluation`)**: Directs execution to `END` if approved or if `max_iteration` is reached; otherwise, routes back to `optimize`.
# 5. **State Reducer (`operator.add`)**: Appends each iteration's draft and feedback into `tweet_history` and `feedback_history` lists without overwriting previous attempts.
