from ai_module import Network, JSON  # type: ignore

def text_to_vector(text: str, length: int) -> list[float]:
    text = text.ljust(length)[:length]
    return [ord(c) / 255.0 for c in text]

def vector_to_text(vector: list[float]) -> str:
    return ''.join(chr(int(round(max(0, min(1, x)) * 255))) for x in vector)

def main():
    INPUT_SIZE = 3
    OUTPUT_SIZE=10
    net = Network([INPUT_SIZE, 5, OUTPUT_SIZE], 0.1, 20000)

    input_text = "Hi!"
    vector_input = text_to_vector(input_text, INPUT_SIZE)
    target_output = text_to_vector("Guten Tag!", OUTPUT_SIZE)

    net.train([vector_input], [target_output])

    output_vector = net.predict(vector_input)
    output_text = vector_to_text(output_vector)

    print("Input Text:", input_text)
    print("Predicted Output Text:", output_text)

if __name__=='__main__':
    main()