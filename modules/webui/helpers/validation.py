"""Request data validation helpers"""

def validate_required_fields(data, required_fields):
    """
    Validate that required fields are present in data

    Args:
        data: Dictionary of data to validate
        required_fields: List of required field names

    Returns:
        tuple: (is_valid, error_message)
    """
    if not data:
        return False, "Request data is required"

    missing_fields = [field for field in required_fields if not data.get(field)]

    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"

    return True, None

def validate_positive_int(value, field_name):
    """Validate that value is a positive integer"""
    try:
        int_value = int(value)
        if int_value <= 0:
            return False, f"{field_name} must be positive"
        return True, None
    except (ValueError, TypeError):
        return False, f"{field_name} must be an integer"

def validate_enum(value, field_name, valid_values):
    """Validate that value is in allowed values"""
    if value not in valid_values:
        return False, f"{field_name} must be one of: {', '.join(valid_values)}"
    return True, None
