def generate_rtl(command):
    template = TEMPLATES.get(command["type"])
    if not template:
        return None
    
    code = template.format(
        bits=command["bits"],
        bits_minus_one=command["bits"] - 1
    )
    return code
