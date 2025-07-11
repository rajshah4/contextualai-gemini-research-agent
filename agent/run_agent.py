from dotenv import load_dotenv
load_dotenv()

from agent.graph import graph
from agent.configuration import Configuration
from agent.state import OverallState

def main():
    user_message = "What was the revenue for nvidia"
    state = OverallState(messages=[user_message])
    config = Configuration()

    print("Running agent...\n")
    last_state = None
    for step in graph.stream(state, config=config.model_dump()):
        print(f"Step keys: {list(step.keys())}")
        for node_name, node_state in step.items():
            print(f"\nNode: {node_name}")
            if isinstance(node_state, dict):
                for k, v in node_state.items():
                    if isinstance(v, list) and len(v) > 5:
                        print(f"  {k}: {v[:3]} ... ({len(v)} items)")
                    else:
                        print(f"  {k}: {v}")
            else:
                print(f"  Output: {node_state}")
            # Save the last node_state for final output
            last_state = node_state
        print("-" * 40)

    # Print the final output using last_state
    if last_state:
        print("\nFinal Output:")
        if "messages" in last_state and last_state["messages"]:
            print(last_state["messages"][0].content)
        else:
            print("No messages found in final state.")
        print("\nSources Gathered:")
        for source in last_state.get("sources_gathered", []):
            print(source)

if __name__ == "__main__":
    main()