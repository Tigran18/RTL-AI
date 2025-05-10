from ai_module import Network, JSON #type: ignore
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

    net = Network([INPUT_SIZE, 64, 64, OUTPUT_SIZE], 0.1, 10000)

    if os.path.exists(MODEL_FILE):
        print(f"Loading model from {MODEL_FILE}...")
        net.load_model(MODEL_FILE)
    else:
        print("Training model from scratch...")
        training_data = [
            ("Create an 8-bit adder",
             '{"module": "adder", "inputs": {"a": {"width": 8}, "b": {"width": 8}}, "outputs": {"sum": {"width": 8}}, "operations": [{"op": "add", "inputs": ["a", "b"], "output": "sum"}]}'),
            ("Make a 1-bit comparator",
             '{"module": "comparator", "inputs": {"a": {"width": 1}, "b": {"width": 1}}, "outputs": {"gt": {"width": 1}}, "operations": [{"op": "gt", "inputs": ["a", "b"], "output": "gt"}]}'),
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
        json_text = vector_to_text(output_vector).strip()

        print("\nPredicted JSON:")
        print(json_text)

        try:
            json_obj = JSON(json_text)
            print("\nGenerated RTL:")
            json_obj.print_rtl()
        except Exception as e:
            print("Error parsing predicted JSON:", e)

if __name__ == "__main__":
    main()
