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
