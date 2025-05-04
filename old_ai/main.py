from parser import parse_input
from generator import generate_rtl
from templates import TEMPLATES, save_templates, load_templates

def main():
    print("\033[96mWelcome to RTL Generator AI! Describe your circuit below. Type 'close' to exit.\033[0m")
    
    while True:
        user_input = input("\nDescribe the circuit you want: ")
        
        if user_input.lower() == 'close':
            break

        command = parse_input(user_input)
        
        if not command:
            print("Sorry, I couldn't understand the request.")
            continue

        rtl_code = generate_rtl(command)
    
        if rtl_code:
            print("\nGenerated RTL code:\n")
            print("\033[92m" + rtl_code + "\033[0m")
        else:
            print(f"Sorry, no template found for type '{command['type']}'.")

            choice = input(f"Do you want to teach me how to generate '{command['type']}'? (yes/no): ").lower()
            if choice == "yes":
                print("Use '{bits}' and '{bits_minus_one}' where needed.")
                new_template = input("Please provide the Verilog template:\n")
                
                global TEMPLATES
                
                TEMPLATES[command["type"]] = new_template
                save_templates(TEMPLATES)
                print("\033[93mThanks! I've learned a new circuit!\033[0m")
                
                TEMPLATES = load_templates()


if __name__ == "__main__":
    main()
