def normalize_name(name):
    """
    Standardizes names for fuzzy string matching by keeping only lowercased alphanumeric characters.
    """
    return "".join(c.lower() for c in name if c.isalnum())


def find_matching_row(condition_name, table_rows):
    """
    Finds a row in the AI-generated explanations table matching the specified condition name.
    """
    norm_cond = normalize_name(condition_name)
    # 1. Exact normalized match
    for row in table_rows:
        if len(row) > 0 and normalize_name(row[0]) == norm_cond:
            return row
    # 2. Substring normalized match
    for row in table_rows:
        if len(row) > 0:
            row_norm = normalize_name(row[0])
            if norm_cond in row_norm or row_norm in norm_cond:
                return row
    return None


def parse_markdown_explanation(markdown_text):
    """
    Parses the AI markdown explanation table into structured condition rows, 
    an executive summary, and a disclaimer footer.
    """
    lines = [line.strip() for line in markdown_text.split("\n") if line.strip()]
    
    table_rows = []
    summary_lines = []
    disclaimer = ""
    
    for line in lines:
        if line.startswith("|"):
            # Skip header divider/separator rows
            if "---" in line:
                continue
            # Skip column title headers
            if "Why It Is Suspected" in line or "Condition" in line:
                continue
            
            parts = [p.strip() for p in line.split("|")]
            # Extract cells between leading and trailing pipes
            if len(parts) >= 6:
                row_data = parts[1:6]
                table_rows.append(row_data)
        else:
            if "assessment is assistive" in line or "healthcare professional" in line:
                disclaimer = line
            else:
                summary_lines.append(line)
                
    summary = "\n\n".join(summary_lines)
    return table_rows, summary, disclaimer
