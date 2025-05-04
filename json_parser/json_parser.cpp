#include "json_parser.hpp"

void JSON::skip_whitespace() {
    while (index < content.size() && std::isspace(content[index])) {
        index++;
    }
}

std::string JSON::parse_string() {
    skip_whitespace();
    if (content[index] != '"') throw std::runtime_error("Expected '\"'");
    index++;
    std::string result;
    while (index < content.size() && content[index] != '"') {
        result += content[index++];
    }
    if (index >= content.size() || content[index] != '"') {
        throw std::runtime_error("Unterminated string");
    }
    index++;
    return result;
}

JSON JSON::parse_object() {
    if (content[index] != '{') throw std::runtime_error("Expected '{'");
    size_t start_index = index;
    JSON nested(content, start_index);
    nested.parse();
    index = nested.index;
    return nested;
}

std::vector<JSONvalues_for_vector> JSON::parse_array() {
    std::vector<JSONvalues_for_vector> arr;
    index++;
    skip_whitespace();
    while (index < content.size() && content[index] != ']') {
        skip_whitespace();
        if (content[index] == '{') {
            arr.push_back(parse_object());
        } else if (content[index] == '"') {
            arr.push_back(parse_string());
        } else if (std::isdigit(content[index]) || content[index] == '-') {
            std::string num;
            if (content[index] == '-') num += content[index++];
            while (index < content.size() && std::isdigit(content[index])) {
                num += content[index++];
            }
            arr.push_back(std::stoi(num));
        } else if (content.compare(index, 4, "true") == 0) {
            index += 4;
            arr.push_back(true);
        } else if (content.compare(index, 5, "false") == 0) {
            index += 5;
            arr.push_back(false);
        } else {
            throw std::runtime_error("Invalid value in array");
        }
        skip_whitespace();
        if (content[index] == ',') {
            index++;
        } else if (content[index] != ']') {
            throw std::runtime_error("Expected ',' or ']'");
        }
        skip_whitespace();
    }
    if (content[index] != ']') {
        throw std::runtime_error("Expected ']'");
    }
    index++;
    return arr;
}

JSONvalue JSON::parse_value() {
    skip_whitespace();
    if (content[index] == '{') return parse_object();
    if (content[index] == '[') return parse_array();
    if (content[index] == '"') return parse_string();
    if (content.compare(index, 4, "true") == 0) {
        index += 4;
        return true;
    }
    if (content.compare(index, 5, "false") == 0) {
        index += 5;
        return false;
    }
    if (content.compare(index, 4, "null") == 0) {
        index += 4;
        return std::string("null");
    }
    std::string result;
    if (content[index] == '-') {
        result += content[index++];
    }
    while (index < content.size() && std::isdigit(content[index])) {
        result += content[index++];
    }
    if (result.empty()) {
        throw std::runtime_error("Invalid value");
    }
    return std::stoi(result);
}

void JSON::parse() {
    skip_whitespace();
    if (content[index] != '{') throw std::runtime_error("Expected '{'");
    index++;
    while (true) {
        skip_whitespace();
        if (content[index] == '}') {
            index++;
            break;
        }
        std::string key = parse_string();
        skip_whitespace();
        if (content[index] != ':') throw std::runtime_error("Expected ':'");
        index++;
        JSONvalue value = parse_value();
        elements[key] = value;
        skip_whitespace();
        if (content[index] == ',') {
            index++;
        } else if (content[index] == '}') {
            index++;
            break;
        } else {
            throw std::runtime_error("Expected ',' or '}'");
        }
    }
}

void JSON::print_value(const JSONvalue& val, int tab = 2) const {
    if (std::holds_alternative<std::string>(val)) {
        std::cout << "[string] " << std::get<std::string>(val);
    } else if (std::holds_alternative<int>(val)) {
        std::cout << "[int] " << std::get<int>(val);
    } else if (std::holds_alternative<bool>(val)) {
        std::cout << "[bool] " << (std::get<bool>(val) ? "true" : "false");
    } else if (std::holds_alternative<JSON>(val)) {
        std::cout << "[object] {\n";
        std::get<JSON>(val).print(tab + 2);
        std::cout << std::string(tab, ' ') << "}";
    } else if (std::holds_alternative<std::vector<JSONvalues_for_vector>>(val)) {
        std::cout << "[array] [ ";
        for (const auto& elem : std::get<std::vector<JSONvalues_for_vector>>(val)) {
            std::visit([&](auto&& arg) {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, std::string>) std::cout << "\"" << arg << "\" ";
                else if constexpr (std::is_same_v<T, int>) std::cout << arg << " ";
                else if constexpr (std::is_same_v<T, bool>) std::cout << (arg ? "true" : "false") << " ";
                else if constexpr (std::is_same_v<T, JSON>) {
                    std::cout << "{ ";
                    arg.print(tab + 2);
                    std::cout << " } ";
                }
            }, elem);
        }
        std::cout << "]";
    }
}

template<typename T>
struct always_false : std::false_type {};

void JSON::print_rtl() const {
    // Lambda definitions for operation handlers
    using OpHandler = std::function<void(const JSON&)>;
    std::map<std::string, OpHandler> handlers;

    // Helper to get string from JSONvalue
    // Accept both JSONvalue and JSONvalues_for_vector
    auto get_str = [](const auto& val) -> std::string {
        using T = std::decay_t<decltype(val)>;
        if constexpr (std::is_same_v<T, JSONvalue>) {
            return std::get<std::string>(val);
        } else if constexpr (std::is_same_v<T, JSONvalues_for_vector>) {
            return std::get<std::string>(val); // assuming it's a string inside the variant
        } else {
            static_assert(always_false<T>::value, "Unsupported type for get_str");
        }
    };


    auto get_inputs = [](const JSON& obj) {
        return std::get<std::vector<JSONvalues_for_vector>>(obj["inputs"]);
    };

    // Handlers for different op types
    handlers["add"] = [&](const JSON& obj) {
        auto in = get_inputs(obj);
        std::cout << "assign " << get_str(obj["output"]) << " = " 
                  << get_str(in[0]) << " + " << get_str(in[1]) << ";\n";
    };

    handlers["sub"] = [&](const JSON& obj) {
        auto in = get_inputs(obj);
        std::cout << "assign " << get_str(obj["output"]) << " = " 
                  << get_str(in[0]) << " - " << get_str(in[1]) << ";\n";
    };

    handlers["and"] = [&](const JSON& obj) {
        auto in = get_inputs(obj);
        std::cout << "assign " << get_str(obj["output"]) << " = " 
                  << get_str(in[0]) << " & " << get_str(in[1]) << ";\n";
    };

    handlers["or"] = [&](const JSON& obj) {
        auto in = get_inputs(obj);
        std::cout << "assign " << get_str(obj["output"]) << " = " 
                  << get_str(in[0]) << " | " << get_str(in[1]) << ";\n";
    };

    handlers["eq"] = [&](const JSON& obj) {
        auto in = get_inputs(obj);
        std::cout << "assign " << get_str(obj["output"]) << " = " 
                  << get_str(in[0]) << " == " << get_str(in[1]) << ";\n";
    };

    handlers["lt"] = [&](const JSON& obj) {
        auto in = get_inputs(obj);
        std::cout << "assign " << get_str(obj["output"]) << " = " 
                  << get_str(in[0]) << " < " << get_str(in[1]) << ";\n";
    };

    handlers["gt"] = [&](const JSON& obj) {
        auto in = get_inputs(obj);
        std::cout << "assign " << get_str(obj["output"]) << " = " 
                  << get_str(in[0]) << " > " << get_str(in[1]) << ";\n";
    };

    handlers["assign"] = [&](const JSON& obj) {
        std::cout << "assign " << get_str(obj["output"]) << " = "
                  << get_str(obj["input"]) << ";\n";
    };

    handlers["mux"] = [&](const JSON& obj) {
        std::string select = get_str(obj["select"]);
        std::string output = get_str(obj["output"]);
        const auto& cases_obj = std::get<JSON>(obj["cases"]);
        std::cout << "always @(*) begin\n";
        std::cout << "  case (" << select << ")\n";
        for (const auto& [key, val] : cases_obj.elements) {
            std::cout << "    " << key << ": " << output << " = " << get_str(val) << ";\n";
        }
        std::cout << "  endcase\n";
        std::cout << "end\n";
    };

    handlers["zero_check"] = [&](const JSON& obj) {
        std::string in = get_str(obj["input"]);
        std::string out = get_str(obj["output"]);
        std::cout << "assign " << out << " = (" << in << " == 0);\n";
    };

    // Emit module declaration
    std::string moduleName = get_str(elements.at("module"));
    const auto& inputs = std::get<JSON>(elements.at("inputs"));
    const auto& outputs = std::get<JSON>(elements.at("outputs"));

    std::cout << "module " << moduleName << " (\n";

    std::vector<std::string> ports;
    for (const auto& [name, val] : inputs.elements)
        ports.push_back(name);
    for (const auto& [name, val] : outputs.elements)
        ports.push_back(name);

    for (size_t i = 0; i < ports.size(); ++i) {
        std::cout << "    " << ports[i];
        if (i != ports.size() - 1) std::cout << ",";
        std::cout << "\n";
    }
    std::cout << ");\n\n";

    // Declare inputs and outputs
    auto declare_ports = [&](const JSON& port_block, const std::string& dir) {
        for (const auto& [name, val] : port_block.elements) {
            const auto& width = std::get<int>(std::get<JSON>(val).elements.at("width"));
            std::cout << dir << " ";
            if (width > 1) std::cout << "[" << (width - 1) << ":0] ";
            std::cout << name << ";\n";
        }
    };

    declare_ports(inputs, "input");
    declare_ports(outputs, "output");

    // Internal wires
    if (elements.count("internal_wires")) {
        const auto& wires = std::get<JSON>(elements.at("internal_wires"));
        for (const auto& [name, val] : wires.elements) {
            int width = std::get<int>(std::get<JSON>(val).elements.at("width"));
            std::cout << "wire ";
            if (width > 1) std::cout << "[" << (width - 1) << ":0] ";
            std::cout << name << ";\n";
        }
    }

    std::cout << "\n";

    // Operations
    if (elements.count("operations")) {
        const auto& ops = std::get<std::vector<JSONvalues_for_vector>>(elements.at("operations"));
        for (const auto& op_val : ops) {
            const auto& op_json = std::get<JSON>(op_val);
            std::string op_type = get_str(op_json["op"]);

            if (handlers.count(op_type)) {
                handlers[op_type](op_json);
            } else {
                std::cerr << "// Warning: Unknown operation '" << op_type << "' skipped.\n";
            }
        }
    }

    std::cout << "endmodule\n";
}



JSON::JSON(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Couldn't open the file\n";
        return;
    }
    char ch;
    while (file.get(ch)) {
        content += ch;
    }
    try {
        parse();
    } catch (const std::exception& e) {
        std::cerr << "JSON parsing error: " << e.what() << "\n";
    }
}

JSON::JSON(const std::string& raw_content, bool isRawString) : content(raw_content), index(0) {
    if (isRawString) {
        try {
            parse();
        } catch (const std::exception& e) {
            std::cerr << "JSON parsing error: " << e.what() << "\n";
        }
    }
}

void JSON::print(int tab = 2) const {
    for (const auto& [key, val] : elements) {
        std::cout << std::string(tab, ' ') << "\"" << key << "\": ";
        print_value(val, tab);
        std::cout << "\n";
    }
}

void JSON::add(const std::string& key, const JSONvalue& val){
    elements[key]=val;
}

const JSONvalue& JSON::operator[](const std::string& key) const {
    return elements.at(key);    
}

std::ostream& operator<<(std::ostream& os, const JSONvalue& val) {
    std::visit([&os](auto&& arg) {
        os << arg;
    }, val);
    return os;
}

std::ostream& operator<<(std::ostream& os, const JSON& json) {
    os << "{ ";
    bool first = true;
    for (const auto& [key, value] : json.elements) {
        if (!first) os << ", ";
        first = false;
        os << "\"" << key << "\": " << value;
    }
    os << " }";
    return os;
}

std::ostream& operator<<(std::ostream& os, const std::vector<JSONvalues_for_vector>& vec) {
    os << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::visit([&os](auto&& arg) {
            os << arg;
        }, vec[i]);

        if (i + 1 < vec.size()) {
            os << ", ";
        }
    }
    os << "]";
    return os;
}
