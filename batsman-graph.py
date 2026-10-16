from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class BatsmanState(TypedDict):
    runs: int
    balls: int
    fours: int
    sixes: int
    strike_rate: float
    bpb: float
    boundary_percentage: float
    summary: str


# Create nodes
def calculate_strike_rate(state: BatsmanState):
    strike_rate = (state['runs'] / state['balls']) * 100
    return {"strike_rate": strike_rate}


def calculate_balls_per_boundary(state: BatsmanState):
    bpb = state["balls"] / (state["fours"] + state["sixes"])
    return {"bpb": bpb}


def calculate_boundary_percent(state: BatsmanState):
    boundary_percent = ((state["fours"] * 4 + state["sixes"] * 6) / state["runs"]) * 100
    return {"boundary_percentage": boundary_percent}


def summary(state: BatsmanState):
    summary_text = (
        f"Strike Rate: {state['strike_rate']}\n"
        f"Balls per Boundary: {state['bpb']}\n"
        f"Boundary Percent: {state['boundary_percentage']}%"
    )
    return {"summary": summary_text}


builder = StateGraph(BatsmanState)
# Add Nodes
builder.add_node("calculate_strike_rate", calculate_strike_rate)
builder.add_node("calculate_balls_per_boundary", calculate_balls_per_boundary)
builder.add_node("calculate_boundary_percent", calculate_boundary_percent)
builder.add_node("summary", summary)

# Add Parallel Edges from START
builder.add_edge(START, "calculate_strike_rate")
builder.add_edge(START, "calculate_balls_per_boundary")
builder.add_edge(START, "calculate_boundary_percent")

# Connect Parallel Nodes to Summary
builder.add_edge("calculate_strike_rate", "summary")
builder.add_edge("calculate_balls_per_boundary", "summary")
builder.add_edge("calculate_boundary_percent", "summary")

# Connect Summary to END
builder.add_edge("summary", END)

# Compile Workflow
graph = builder.compile()

# Execute Workflow
initial_state = {
    "runs": 100,
    "balls": 50,
    "fours": 6,
    "sixes": 4
}

result = graph.invoke(initial_state)
print(result)





