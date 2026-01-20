from agents1 import create_conversational_graph

def test_initial_greeting_flow():
    print("Testing initial graph flow (Greeting -> Empty Input)...")
    
    graph = create_conversational_graph()
    
    initial_state = {
        "conversation_stage": "greeting",
        "messages": [],
        "user_input": "", # Empty input simulating start
        "intent": "",
        "user_profile": {},
        "response": "",
        "context": {}
    }
    
    # Invoke the graph
    result = graph.invoke(initial_state)
    
    print(f"\nFinal Stage: {result.get('conversation_stage')}")
    print(f"Intent detected: {result.get('intent')}")
    print(f"Response: {result.get('response')[:]}...")
    
    # Verification
    if result.get("intent") == "waiting":
        print("\n✅ SUCCESS: Graph waited for input correctly.")
    elif result.get("intent") == "unclear":
        print("\n❌ FAILURE: Graph detected 'unclear' prematurely.")
    else:
        print(f"\n⚠️ UNEXPECTED: {result.get('intent')}")

if __name__ == "__main__":
    test_initial_greeting_flow()
