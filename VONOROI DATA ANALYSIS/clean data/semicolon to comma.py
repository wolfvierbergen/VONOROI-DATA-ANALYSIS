def replace_semicolon_with_comma(input_file, output_file):
    with open(input_file, 'r') as file:
        content = file.read()
        content = content.replace(';', ',') #replace semicolon with comma
    
    with open(output_file, 'w') as file:
        file.write(content)

replace_semicolon_with_comma('original dataset.csv', 'dataset no semicolon.csv')