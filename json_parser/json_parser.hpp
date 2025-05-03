#pragma once
#include <map>
#include <string>
#include <iostream>
#include <fstream>
#include <cctype>
#include <stdexcept>
#include <variant>
#include <vector>

class JSON;

using JSONvalues_for_vector = std::variant<std::string, int, bool, JSON>;
using JSONvalue = std::variant<std::string, int, bool, JSON, std::vector<JSONvalues_for_vector>>;

class JSON {
private:
    std::map<std::string, JSONvalue> elements;
    std::string content;
    size_t index = 0;

    void skip_whitespace();
    std::string parse_string();
    JSON parse_object();
    std::vector<JSONvalues_for_vector> parse_array();
    JSONvalue parse_value();
    void parse();
    void print_value(const JSONvalue& val, int tab) const;

    JSON(const std::string& input, size_t startIndex) : content(input), index(startIndex) {}

public:
    JSON(const std::string& filename);
    JSON(const std::string& raw_content, bool isRawString);
    
    void print(int tab) const;
    void add(const std::string& key, const JSONvalue& val);

    const JSONvalue& operator[](const std::string& key) const;

    friend std::ostream& operator<<(std::ostream& os, const JSON& json);
};

std::ostream& operator<<(std::ostream& os, const JSONvalue& val);
std::ostream& operator<<(std::ostream& os, const std::vector<JSONvalues_for_vector>& vec);
