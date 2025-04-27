import re

def parse_input(user_input):
    user_input = user_input.lower()

    numbers = re.findall(r'\d+', user_input)
    if not numbers:
        return None
    bits = int(numbers[0])

    if "multiplexer" in user_input:
        return {"type": "multiplexer", "bits": bits}
    
    if "adder" in user_input:
        return {"type": "adder", "bits": bits}
    
    words = user_input.split()
    for word in words:
        if "bit" in word:
            continue
        if word.isalpha() and word not in ["create", "make", "build", "design"]:
            return {"type": word, "bits": bits}
    
    return None
