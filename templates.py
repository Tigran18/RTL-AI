import json
import os

TEMPLATES_FILE = "templates.json"

TEMPLATES = {
    "multiplexer": """
module mux_{bits}bit(input [{bits_minus_one}:0] A, input [{bits_minus_one}:0] B, input SEL, output [{bits_minus_one}:0] Y);
    assign Y = SEL ? B : A;
endmodule
""",
    "adder": """
module adder_{bits}bit(input [{bits_minus_one}:0] A, input [{bits_minus_one}:0] B, output [{bits_minus_one}:0] SUM);
    assign SUM = A + B;
endmodule
"""
}

def save_templates(templates):
    with open(TEMPLATES_FILE, "w") as f:
        json.dump(templates, f)

def load_templates():
    if os.path.exists(TEMPLATES_FILE):
        with open(TEMPLATES_FILE, "r") as f:
            return json.load(f)
    return TEMPLATES

TEMPLATES = load_templates()
