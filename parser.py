import re

def parse_input(user_input):
    user_input = user_input.lower()

    # Find first number in the text
    numbers = re.findall(r'\d+', user_input)
    if not numbers:
        return None
    bits = int(numbers[0])

    if "multiplexer" in user_input:
        return {"type": "multiplexer", "bits": bits}
    
    if "adder" in user_input:
        return {"type": "adder", "bits": bits}
    
    return None
