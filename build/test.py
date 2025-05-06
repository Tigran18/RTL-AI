from ai_module import Network, JSON  # type: ignore

def text_to_vector(text: str, length: int) -> list[float]:
    text = text.ljust(length)[:length]
    return [ord(c) / 255.0 for c in text]

def vector_to_text(vector: list[float]) -> str:
    return ''.join(chr(int(round(max(0, min(1, x)) * 255))) for x in vector)

def main():
    INPUT_SIZE = 40
    OUTPUT_SIZE = 60
    net = Network([INPUT_SIZE, 40, 40, OUTPUT_SIZE], 0.1, 100000)

    training_data = [
        ("Hi!", "Hello! How can I help you today?"),
        ("How are you?", "I'm doing great, thank you for asking! How about you?"),
        ("What is your name?", "I am an AI assistant, here to help you with anything you need."),
        ("What do you do?", "I can assist with programming, answer questions, and more! What can I do for you?"),
        ("Can you help me with Python?", "Absolutely! I can assist you with Python coding. What do you need help with?"),
        ("Tell me a joke.", "Why don't skeletons fight each other? They don't have the guts!"),
        ("What is the capital of France?", "The capital of France is Paris."),
        ("Who is the president of the United States?", "The president of the United States is Joe Biden."),
        ("What is 2 + 2?", "2 + 2 equals 4."),
        ("What is the weather like?", "I don't have access to weather data, but you can check your local forecast!"),
        ("What is AI?", "AI stands for Artificial Intelligence, which involves machines designed to perform tasks that typically require human intelligence."),
        ("Can you write code for me?", "Of course! What kind of code would you like me to write?"),
        ("What is 10 * 10?", "10 * 10 equals 100."),
        ("Do you like music?", "I don't have preferences, but I know a lot about music! What genre do you like?"),
        ("Tell me a fun fact.", "Did you know that honey never spoils? Archaeologists have found pots of honey in ancient tombs that are over 3,000 years old!"),
        ("What is your favorite color?", "I don't have a favorite color, but I can tell you about any color you like!"),
        ("How tall is Mount Everest?", "Mount Everest is 8,848.86 meters tall."),
        ("What is the square root of 64?", "The square root of 64 is 8."),
        ("Can you solve puzzles?", "I can certainly try! Do you have a puzzle you'd like me to solve?"),
        ("What is your favorite movie?", "I don't watch movies, but I can recommend some based on your preferences! What type of movie do you like?"),
        ("Where is the Great Wall of China?", "The Great Wall of China is in northern China and stretches over 13,000 miles."),
        ("How do you work?", "I process input and use trained data to generate an output. My responses are based on patterns I have learned from various text data."),
        ("Can you learn new things?", "I can improve my responses through more training, but I don't learn in real-time from individual conversations."),
        ("What is a black hole?", "A black hole is a region of space where gravity is so strong that nothing, not even light, can escape it."),
        ("Do you like to chat?", "I do! I'm here to have conversations and help out in any way I can."),
        ("What is the meaning of life?", "The meaning of life is a philosophical question. Some believe it’s about finding happiness, others think it’s about purpose. What do you think?"),
        ("Can you translate languages?", "Yes, I can help translate various languages. What language would you like to translate to or from?"),
        ("What is the time?", "I don't have access to real-time data, but you can easily check the time on your device!"),
        ("What is a computer virus?", "A computer virus is a type of malicious software that can replicate itself and spread to other computers, often causing harm."),
        ("Tell me a fun programming fact.", "Did you know that the first computer programmer was Ada Lovelace? She wrote algorithms for Charles Babbage's Analytical Engine."),
        ("What is your favorite book?", "I don't read books, but I can recommend some! What type of books do you enjoy?"),
        ("What is the largest planet in our solar system?", "The largest planet in our solar system is Jupiter."),
        ("Can you calculate factorials?", "Yes, I can! For example, the factorial of 5 is 5 * 4 * 3 * 2 * 1 = 120."),
        ("What is the fastest animal on Earth?", "The fastest animal on Earth is the peregrine falcon, which can reach speeds of over 240 mph when diving."),
        ("Can you help me with math?", "Of course! What kind of math problem do you need help with?"),
        ("What is the difference between a cat and a dog?", "Cats are generally more independent, while dogs are known for being loyal and affectionate. Do you have a preference?"),
        ("How do you make a cup of coffee?", "To make a basic cup of coffee, brew ground coffee with hot water, then enjoy! Do you prefer any special flavoring?"),
        ("What is the longest river in the world?", "The longest river in the world is the Nile River, measuring about 6,650 kilometers."),
        ("What is a tornado?", "A tornado is a violently rotating column of air extending from a thunderstorm to the ground, capable of causing significant damage."),
        ("What are your hobbies?", "I don't have hobbies, but I can talk about a wide range of interests! What hobbies do you enjoy?"),
        ("What is 3 + 7?", "3 + 7 equals 10."),
    ]

    inputs = [text_to_vector(q, INPUT_SIZE) for q, _ in training_data]
    targets = [text_to_vector(a, OUTPUT_SIZE) for _, a in training_data]

    net.train(inputs, targets)

    print("Training complete. You can now chat with the AI!")

    while True:
        user_input = input("You: ")

        input_vector = text_to_vector(user_input, INPUT_SIZE)
        output_vector = net.predict(input_vector)
        ai_response = vector_to_text(output_vector).strip()

        print("AI:", ai_response)

        if user_input.lower() in ("bye", "exit", "quit"):
            break

if __name__=='__main__':
    main()