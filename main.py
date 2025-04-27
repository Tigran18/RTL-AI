from parser import parse_input
from generator import generate_rtl
from templates import TEMPLATES

def main():
    user_input = input("Describe the circuit you want: ")
    command = parse_input(user_input)
    
    if not command:
        print("Sorry, I couldn't understand the request.")
        return
    
    rtl_code = generate_rtl(command)
    
    if rtl_code:
        print("\nGenerated RTL code:\n")
        print(rtl_code)
    else:
        print("Sorry, no template found.")

if __name__ == "__main__":
    main()
