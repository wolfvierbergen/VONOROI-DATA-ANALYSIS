def replace_semicolon_with_comma(input_file, output_file):
    with open(input_file, 'r') as file:
        content = file.read()
        content = content.replace(',', ';') #replace comma with semicolon
    
    with open(output_file, 'w') as file:
        file.write(content)

replace_semicolon_with_comma('processed_output.csv', 'processed_output yes semicolon.csv')