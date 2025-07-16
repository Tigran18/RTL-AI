from ai_module import Network, ActivationType, JSON  # type: ignore
import os

MODEL_FILE = "network_weights.txt"

def text_to_vector(text: str, length: int) -> list[float]:
    text = text.ljust(length)[:length]
    return [ord(c) / 255.0 for c in text]

def vector_to_text(vector: list[float]) -> str:
    return ''.join(chr(int(round(max(0, min(1, x)) * 255))) for x in vector)

def main():
    INPUT_SIZE = 40
    OUTPUT_SIZE = 300

    activations = [ActivationType.ReLU] * 4  # same length as layer sizes
    net = Network([INPUT_SIZE, 64, 64, OUTPUT_SIZE], activations, 0.1, 10000, 32, 0.0)

    if os.path.exists(MODEL_FILE):
        print(f"Loading model from {MODEL_FILE}...")
        net.load_model(MODEL_FILE)
    else:
        print("Training model from scratch...")
        training_data = [
            # Original examples
            ("Hey!", "Hello!"),
            ("Create an 8-bit adder",
             '{"module": "adder", "inputs": {"a": {"width": 8}, "b": {"width": 8}}, "outputs": {"sum": {"width": 8}}, "operations": [{"op": "add", "inputs": ["a", "b"], "output": "sum"}]}'),
            ("Make a 1-bit comparator",
             '{"module": "comparator", "inputs": {"a": {"width": 1}, "b": {"width": 1}}, "outputs": {"gt": {"width": 1}}, "operations": [{"op": "gt", "inputs": ["a", "b"], "output": "gt"}]}'),
            # New examples with human-like conversational prompts
            ("I need a 4-bit counter",
             '{"module": "counter", "inputs": {"clk": {"width": 1}, "rst": {"width": 1}}, "outputs": {"count": {"width": 4}}, "operations": [{"op": "increment", "inputs": ["count"], "output": "count", "trigger": "clk", "reset": "rst"}]}'),
            ("Build a 2-to-1 mux 16-bit",
             '{"module": "mux", "inputs": {"a": {"width": 16}, "b": {"width": 16}, "sel": {"width": 1}}, "outputs": {"out": {"width": 16}}, "operations": [{"op": "mux", "select": "sel", "cases": {"0": "a", "1": "b"}, "output": "out"}]}'),
            ("Design a 2-bit AND gate",
             '{"module": "and_gate", "inputs": {"a": {"width": 2}, "b": {"width": 2}}, "outputs": {"out": {"width": 2}}, "operations": [{"op": "and", "inputs": ["a", "b"], "output": "out"}]}'),
            ("Give me a 1-bit flip-flop",
             '{"module": "flipflop", "inputs": {"d": {"width": 1}, "clk": {"width": 1}}, "outputs": {"q": {"width": 1}}, "operations": [{"op": "latch", "input": "d", "output": "q", "trigger": "clk"}]}'),
            ("Can you make a 4-bit shifter?",
             '{"module": "shifter", "inputs": {"data": {"width": 4}, "shift": {"width": 2}}, "outputs": {"out": {"width": 4}}, "operations": [{"op": "shift", "input": "data", "amount": "shift", "output": "out"}]}'),
            ("I want an 8-bit OR circuit",
             '{"module": "or_gate", "inputs": {"a": {"width": 8}, "b": {"width": 8}}, "outputs": {"out": {"width": 8}}, "operations": [{"op": "or", "inputs": ["a", "b"], "output": "out"}]}'),
            ("Build a 3-input XOR gate",
             '{"module": "xor_gate", "inputs": {"a": {"width": 1}, "b": {"width": 1}, "c": {"width": 1}}, "outputs": {"out": {"width": 1}}, "operations": [{"op": "xor", "inputs": ["a", "b", "c"], "output": "out"}]}'),
            ("Please create a 16-bit adder",
             '{"module": "adder", "inputs": {"a": {"width": 16}, "b": {"width": 16}}, "outputs": {"sum": {"width": 16}}, "operations": [{"op": "add", "inputs": ["a", "b"], "output": "sum"}]}'),
            # More conversational examples
            ("Hey, I need a simple mux",
             '{"module": "mux", "inputs": {"a": {"width": 8}, "b": {"width": 8}, "sel": {"width": 1}}, "outputs": {"out": {"width": 8}}, "operations": [{"op": "mux", "select": "sel", "cases": {"0": "a", "1": "b"}, "output": "out"}]}'),
            ("Can you do a 2-bit counter?",
             '{"module": "counter", "inputs": {"clk": {"width": 1}, "rst": {"width": 1}}, "outputs": {"count": {"width": 2}}, "operations": [{"op": "increment", "inputs": ["count"], "output": "count", "trigger": "clk", "reset": "rst"}]}'),
            ("Make me a 4-bit subtractor",
             '{"module": "subtractor", "inputs": {"a": {"width": 4}, "b": {"width": 4}}, "outputs": {"diff": {"width": 4}}, "operations": [{"op": "subtract", "inputs": ["a", "b"], "output": "diff"}]}'),
            ("I’d like a 1-bit NOT gate",
             '{"module": "not_gate", "inputs": {"a": {"width": 1}}, "outputs": {"out": {"width": 1}}, "operations": [{"op": "not", "input": "a", "output": "out"}]}'),
            ("Design a 32-bit adder please",
             '{"module": "adder", "inputs": {"a": {"width": 32}, "b": {"width": 32}}, "outputs": {"sum": {"width": 32}}, "operations": [{"op": "add", "inputs": ["a", "b"], "output": "sum"}]}'),
        ]

        inputs = [text_to_vector(q, INPUT_SIZE) for q, _ in training_data]
        targets = [text_to_vector(a, OUTPUT_SIZE) for _, a in training_data]

        net.train(inputs, targets)
        net.save_model(MODEL_FILE)
        print(f"Model saved to {MODEL_FILE}")

    print("\nReady to generate RTL from text prompts.")
    while True:
        user_input = input("Request (or 'exit'): ")
        if user_input.lower() in ("exit", "quit"):
            break

        input_vector = text_to_vector(user_input, INPUT_SIZE)
        output_vector = net.predict(input_vector)
        json_text = vector_to_text(output_vector).rstrip("\x00").strip()

        if json_text.lstrip().startswith(("{", "[")):
            try:
                json_obj = JSON(json_text, True)
                print("\nGenerated RTL:")
                json_obj.print_rtl()
            except Exception as e:
                print("Error parsing predicted JSON:", e)
        else:
            print("\nModel says:", json_text)


if __name__ == "__main__":
    main()